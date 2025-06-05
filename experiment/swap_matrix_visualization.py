import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

# Assuming rwkv_model.py and config.py are in the same directory or accessible in PYTHONPATH
from rwkv_model import RWKV7_Model_Classifier, RWKV_TimeMix

try:
    from config import (DATASET_FILE_CONFIG, MODEL_CHECKPOINT_PATH_CONFIG,
                        MODEL_HYPERPARAMETERS, PAD_TOKEN_CONFIG)
except ImportError:
    print("Warning: config.py not found or variables not defined. Using default values for demonstration.")
    MODEL_HYPERPARAMETERS = {
        "D_MODEL": 8, "N_LAYER": 4, "HEAD_SIZE": 8, "FFN_HIDDEN_MULTIPLIER": 4,
        "LORA_DIM_W": 32, "LORA_DIM_A": 32, "LORA_DIM_V": 16, "LORA_DIM_G": 32
    }
    DATASET_FILE_CONFIG = 'regex_dataset.json'
    MODEL_CHECKPOINT_PATH_CONFIG = 'best_rwkv_regex_model.pth'
    PAD_TOKEN_CONFIG = '<pad>'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility Functions ---
def load_vocab_and_config(dataset_file_path):
    try:
        with open(dataset_file_path, 'r') as f:
            dataset_obj = json.load(f)
        vocab = dataset_obj['vocab']
        vocab_size = len(vocab)
        target_substring = dataset_obj.get('target_substring', "abbccc")
        return vocab, vocab_size, target_substring
    except FileNotFoundError:
        print(f"Error: Dataset file {dataset_file_path} not found.")
        return None, 0, ""
    except Exception as e:
        print(f"Error loading dataset config from {dataset_file_path}: {e}")
        return None, 0, ""

def create_theoretical_swap_matrix_c2(size, idx1, idx2):
    """Creates the theoretical swap matrix as per Lemma 1 (using c=2)."""
    if not (0 <= idx1 < size and 0 <= idx2 < size):
        raise ValueError("Indices are out of bounds for the given matrix size.")
    if idx1 == idx2:
        return np.identity(size)
    identity = np.identity(size)
    e_x = np.zeros(size); e_x[idx1] = 1.0
    e_y = np.zeros(size); e_y[idx2] = 1.0
    diff_vec_row = (e_x - e_y)
    rank_one_term = np.outer(diff_vec_row.reshape(-1, 1), diff_vec_row)
    swap_matrix_c2 = identity - rank_one_term
    return swap_matrix_c2

def get_learned_time_mix_params_for_input(
    model, 
    current_token_id, 
    prev_token_id, 
    layer_id_to_inspect,
    vocab_map
    ):
    """
    Performs a partial forward pass to get the learned w_t, a_t, and kappa_hat_t
    for a specific input token, previous token context, and layer.
    Returns these parameters for ALL heads in the specified layer.
    """
    head_size = model.head_size
    num_heads = model.num_heads

    current_token_tensor = torch.tensor([[current_token_id]], device=DEVICE, dtype=torch.long)
    prev_token_tensor = torch.tensor([[prev_token_id]], device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        x_emb_current = model.embedding(current_token_tensor)
        x_emb_prev = model.embedding(prev_token_tensor)
        
        if layer_id_to_inspect == 0:
            x_input_to_block_ln = model.ln_pre_blocks(x_emb_current)
        else:
            x_input_to_block_ln = x_emb_current

        time_mix_block = model.blocks[layer_id_to_inspect].tm
        x_for_tm_params = model.blocks[layer_id_to_inspect].ln_tm_in(x_input_to_block_ln)
        shift_state_for_tm_params = x_emb_prev

        B, T, _ = x_for_tm_params.shape 
        
        x_k_lerp = x_for_tm_params + (shift_state_for_tm_params - x_for_tm_params) * time_mix_block.mu_k
        x_d_lerp = x_for_tm_params + (shift_state_for_tm_params - x_for_tm_params) * time_mix_block.mu_d
        x_a_lerp = x_for_tm_params + (shift_state_for_tm_params - x_for_tm_params) * time_mix_block.mu_a
        
        k_vec = time_mix_block.W_k(x_k_lerp)
        d_lora_out = time_mix_block.decay_lora(x_d_lerp) 
        w_vec = torch.exp(-torch.exp(torch.tensor(-0.5, device=DEVICE, dtype=torch.float32)) * torch.sigmoid(d_lora_out.float())).type_as(x_for_tm_params)
        a_vec = torch.sigmoid(time_mix_block.iclr_lora(x_a_lerp).float()).type_as(x_for_tm_params)
        kappa_vec = k_vec * time_mix_block.removal_key_multiplier_xi 

        w_head_all = w_vec.view(B, T, num_heads, head_size)
        a_head_all = a_vec.view(B, T, num_heads, head_size)
        kappa_head_all = kappa_vec.view(B, T, num_heads, head_size)

        kappa_hat_head_all = F.normalize(kappa_head_all, p=2, dim=-1)

        w_t_all_heads = w_head_all.squeeze(0).squeeze(0)
        a_t_all_heads = a_head_all.squeeze(0).squeeze(0)
        kappa_hat_t_all_heads = kappa_hat_head_all.squeeze(0).squeeze(0)

    return w_t_all_heads.cpu().numpy(), \
           a_t_all_heads.cpu().numpy(), \
           kappa_hat_t_all_heads.cpu().numpy()


def calculate_actual_Gt_from_learned_params(w_t_head, a_t_head, kappa_hat_t_head):
    """
    Calculates the actual G_t for a single head using its learned parameters.
    The model's formula is G_t = diag(w_t) - kappa_hat_t^T @ (a_t . kappa_hat_t)
    """
    diag_wt = np.diag(w_t_head)
    term_a_kappa_hat = a_t_head * kappa_hat_t_head
    kappa_hat_col = kappa_hat_t_head.reshape(-1, 1)
    term_a_kappa_hat_row = term_a_kappa_hat.reshape(1, -1)
    outer_product_term = kappa_hat_col @ term_a_kappa_hat_row
    Gt_actual = diag_wt - outer_product_term
    return Gt_actual, diag_wt, outer_product_term

# --- Main Execution ---
def main():
    print(f"Using device: {DEVICE}")

    print("\n--- Loading Model and Configuration ---")
    vocab, vocab_size, _ = load_vocab_and_config(DATASET_FILE_CONFIG)
    
    if vocab is None:
        print("Exiting due to vocabulary loading failure.")
        return
    
    model_checkpoint_path = MODEL_CHECKPOINT_PATH_CONFIG
    
    model = RWKV7_Model_Classifier(
        d_model=MODEL_HYPERPARAMETERS["D_MODEL"],
        n_layer=MODEL_HYPERPARAMETERS["N_LAYER"],
        vocab_size=vocab_size,
        head_size=MODEL_HYPERPARAMETERS["HEAD_SIZE"],
        ffn_hidden_multiplier=MODEL_HYPERPARAMETERS["FFN_HIDDEN_MULTIPLIER"],
        lora_dim_w=MODEL_HYPERPARAMETERS["LORA_DIM_W"],
        lora_dim_a=MODEL_HYPERPARAMETERS["LORA_DIM_A"],
        lora_dim_v=MODEL_HYPERPARAMETERS["LORA_DIM_V"],
        lora_dim_g=MODEL_HYPERPARAMETERS["LORA_DIM_G"]
    ).to(DEVICE)

    if os.path.exists(model_checkpoint_path):
        try:
            model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE))
            print(f"Successfully loaded trained model weights from {model_checkpoint_path}")
        except Exception as e:
            print(f"Could not load trained model weights from {model_checkpoint_path}: {e}")
            print("Exiting. This script requires the trained model.")
            return
    else:
        print(f"Trained model checkpoint {model_checkpoint_path} not found. Exiting.")
        return
    model.eval()

    head_size = MODEL_HYPERPARAMETERS["HEAD_SIZE"]
    head_to_inspect = 0
    
    test_cases = [
        {'current_char': 'a', 'prev_char': 'b', 'desc': "Input 'a', prev 'b'"},
        {'current_char': 'b', 'prev_char': 'a', 'desc': "Input 'b', prev 'a'"},
        {'current_char': 'c', 'prev_char': 'a', 'desc': "Input 'c', prev 'a'"},
        {'current_char': 'a', 'prev_char': 'c', 'desc': "Input 'a', prev 'c'"},
        {'current_char': 'b', 'prev_char': 'c', 'desc': "Input 'b', prev 'c'"},
        {'current_char': 'c', 'prev_char': 'b', 'desc': "Input 'c', prev 'b'"},
    ]

    idx_to_swap_1 = 0
    idx_to_swap_2 = 1
    if head_size <= 1: 
        idx_to_swap_2 = 0
        
    theoretical_swap_c2_matrix = create_theoretical_swap_matrix_c2(head_size, idx_to_swap_1, idx_to_swap_2)
    print(f"\n--- Theoretical Swap Matrix (c=2, swaps idx {idx_to_swap_1} & {idx_to_swap_2}) ---")
    print(np.round(theoretical_swap_c2_matrix, decimals=3))
    print("-" * 50)

    # Layers to inspect (0-4 or up to available layers)
    layers_to_inspect = list(range(min(5, MODEL_HYPERPARAMETERS["N_LAYER"])))
    
    for case_idx, case in enumerate(test_cases):
        current_char_input = case['current_char']
        prev_char_input = case['prev_char']
        description = case['desc']

        current_token_id = vocab.get(current_char_input)
        prev_token_id = vocab.get(prev_char_input)

        if current_token_id is None or prev_token_id is None:
            print(f"Error: Token(s) '{current_char_input}' or '{prev_char_input}' not in vocab. Skipping case.")
            continue

        print(f"\n--- Creating Visualization for: {description} ---")
        
        # Create figure with subplots for this test case (5 rows for all components)
        fig, axes = plt.subplots(5, len(layers_to_inspect), figsize=(4*len(layers_to_inspect), 20))
        if len(layers_to_inspect) == 1:
            axes = axes.reshape(-1, 1)
        
        for layer_idx, layer_to_inspect in enumerate(layers_to_inspect):
            try:
                learned_w_all_h, learned_a_all_h, learned_kappa_hat_all_h = \
                    get_learned_time_mix_params_for_input(model, current_token_id, prev_token_id, layer_to_inspect, vocab)
                
                learned_w_head = learned_w_all_h[head_to_inspect]
                learned_a_head = learned_a_all_h[head_to_inspect]
                learned_kappa_hat_head = learned_kappa_hat_all_h[head_to_inspect]

                actual_Gt_matrix, diag_wt_matrix, outer_product_matrix = calculate_actual_Gt_from_learned_params(
                    learned_w_head, learned_a_head, learned_kappa_hat_head
                )
                
                # Plot theoretical matrix (row 0)
                sns.heatmap(theoretical_swap_c2_matrix, 
                           annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                           ax=axes[0, layer_idx], cbar=layer_idx==len(layers_to_inspect)-1,
                           vmin=-1, vmax=1)
                axes[0, layer_idx].set_title(f'Theoretical\nLayer {layer_to_inspect}')
                
                # Plot diag(w_t) matrix (row 1)
                sns.heatmap(diag_wt_matrix, 
                           annot=True, fmt='.2f', cmap='viridis',
                           ax=axes[1, layer_idx], cbar=layer_idx==len(layers_to_inspect)-1)
                axes[1, layer_idx].set_title(f'diag(w_t)\nLayer {layer_to_inspect}')
                
                # Plot outer product term (row 2)
                sns.heatmap(outer_product_matrix, 
                           annot=True, fmt='.2f', cmap='plasma',
                           ax=axes[2, layer_idx], cbar=layer_idx==len(layers_to_inspect)-1)
                axes[2, layer_idx].set_title(f'Outer Product Term\nLayer {layer_to_inspect}')
                
                # Plot experimental G_t matrix (row 3)
                sns.heatmap(actual_Gt_matrix, 
                           annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                           ax=axes[3, layer_idx], cbar=layer_idx==len(layers_to_inspect)-1,
                           vmin=-1, vmax=1)
                axes[3, layer_idx].set_title(f'Experimental G_t\nLayer {layer_to_inspect}')
                
                # Plot difference matrix (row 4)
                diff_matrix = actual_Gt_matrix - theoretical_swap_c2_matrix
                sns.heatmap(diff_matrix, 
                           annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                           ax=axes[4, layer_idx], cbar=layer_idx==len(layers_to_inspect)-1)
                axes[4, layer_idx].set_title(f'Difference\nLayer {layer_to_inspect}')
                
                # Calculate similarity metrics
                diff_norm = np.linalg.norm(diff_matrix)
                print(f"Layer {layer_to_inspect}: Norm difference = {diff_norm:.4f}")
                
            except Exception as e:
                print(f"Error processing layer {layer_to_inspect}: {e}")
                for row in range(5):
                    axes[row, layer_idx].text(0.5, 0.5, 'ERROR', ha='center', va='center', 
                                            transform=axes[row, layer_idx].transAxes)
        
        plt.suptitle(f'G_t Matrix Components Analysis | '
                     f'Input: {current_char_input}, Previous: {prev_char_input}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        filename = f'Gt_components_analysis_case_{case_idx+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved detailed visualization as {filename}")
        plt.close()  # Close the figure to free memory
        
    print("\n--- Summary Statistics Across All Cases ---")
    all_differences = []
    
    for case in test_cases:
        current_char_input = case['current_char']
        prev_char_input = case['prev_char']
        
        current_token_id = vocab.get(current_char_input)
        prev_token_id = vocab.get(prev_char_input)
        
        if current_token_id is None or prev_token_id is None:
            continue
            
        for layer_to_inspect in layers_to_inspect:
            try:
                learned_w_all_h, learned_a_all_h, learned_kappa_hat_all_h = \
                    get_learned_time_mix_params_for_input(model, current_token_id, prev_token_id, layer_to_inspect, vocab)
                
                learned_w_head = learned_w_all_h[head_to_inspect]
                learned_a_head = learned_a_all_h[head_to_inspect]
                learned_kappa_hat_head = learned_kappa_hat_all_h[head_to_inspect]

                actual_Gt_matrix, _, _ = calculate_actual_Gt_from_learned_params(
                    learned_w_head, learned_a_head, learned_kappa_hat_head
                )
                
                diff_norm = np.linalg.norm(actual_Gt_matrix - theoretical_swap_c2_matrix)
                all_differences.append(diff_norm)
                
            except Exception:
                continue
    
    if all_differences:
        print(f"Mean difference norm: {np.mean(all_differences):.4f}")
        print(f"Std difference norm: {np.std(all_differences):.4f}")
        print(f"Min difference norm: {np.min(all_differences):.4f}")
        print(f"Max difference norm: {np.max(all_differences):.4f}")

    print("\nOverall Conclusion:")
    print("This script compares the theoretical swap matrix G_t with the experimental")
    print("matrices learned by the RWKV model across different layers and input contexts.")
    print("The detailed component analysis shows:")
    print("- diag(w_t): The diagonal decay component")
    print("- Outer Product Term: The removal mechanism κ̂_t^T @ (a_t ⊙ κ̂_t)")
    print("- G_t = diag(w_t) - outer_product_term")
    print("- Difference from theoretical ideal")

if __name__ == "__main__":
    main()