import json
import os

import numpy as np
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
        target_substring = dataset_obj.get('target_substring', "abbccc") # For context
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
    vocab_map # Pass vocab_map to get PAD_TOKEN_ID
    ):
    """
    Performs a partial forward pass to get the learned w_t, a_t, and kappa_hat_t
    for a specific input token, previous token context, and layer.
    Returns these parameters for ALL heads in the specified layer.
    """
    d_model = model.d_model
    head_size = model.head_size
    num_heads = model.num_heads

    # --- Prepare inputs for the TimeMix block ---
    # 1. Embedding for current and previous token
    # We need x_t (current_x_for_tm) and x_{t-1} (shift_state_for_tm)
    # Both will be (1, 1, D_MODEL)
    
    # current_token_id is an int, prev_token_id is an int
    current_token_tensor = torch.tensor([[current_token_id]], device=DEVICE, dtype=torch.long)
    prev_token_tensor = torch.tensor([[prev_token_id]], device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        # Get embeddings
        x_emb_current = model.embedding(current_token_tensor) # (1, 1, D_MODEL)
        x_emb_prev = model.embedding(prev_token_tensor)       # (1, 1, D_MODEL)
        
        # LayerNorm before blocks (applied to current token's embedding)
        # For layer 0, the input 'x' to RWKV_Block is ln_pre_blocks(x_emb_current)
        # For subsequent layers, it's the output of the previous block.
        # For simplicity in extracting params for ONE layer's TimeMix,
        # we'll simulate the input 'x' to that TimeMix block.
        # Let's assume for this focused test, the input to the target TimeMix block's LayerNorm
        # is the current token's raw embedding. This is a simplification.
        # A more rigorous way would be to pass through all preceding layers.
        # However, the TimeMix parameters (w,a,k) are derived from x_t and x_{t-1}
        # *after* their respective LayerNorms and token shifts.

        # The input to RWKV_Block is 'current_x'
        # The input to TimeMix's internal LayerNorm is this 'current_x'.
        # Let's use x_emb_current as the 'x' that goes into the target block's ln_tm_in
        # This is an approximation if layer_id_to_inspect > 0.
        # For layer_id_to_inspect = 0, x = model.ln_pre_blocks(x_emb_current)
        
        if layer_id_to_inspect == 0:
            x_input_to_block_ln = model.ln_pre_blocks(x_emb_current)
        else:
            # This is a simplification: ideally, we'd propagate through previous layers.
            # For this test, we'll use the current embedding as a proxy for the input
            # to the target layer's TimeMix LayerNorm.
            # This focuses on how *this layer's* TimeMix processes an input.
            print(f"Warning: For layer {layer_id_to_inspect}, using current token embedding as direct input to its TimeMix LN for simplicity.")
            x_input_to_block_ln = x_emb_current # Simplified input for higher layers

        # The specific TimeMix block
        time_mix_block = model.blocks[layer_id_to_inspect].tm
        
        # Input to TimeMix is LayerNorm(x_input_to_block_ln)
        x_for_tm_params = model.blocks[layer_id_to_inspect].ln_tm_in(x_input_to_block_ln) # (1,1,D_MODEL)
        
        # shift_state_prev for TimeMix parameter calculation corresponds to x_{t-1}
        # This x_{t-1} would also have been processed by its own LayerNorm in the previous step.
        # For simplicity, let's use x_emb_prev as the representation of x_{t-1}
        # This is x_shifted in the TimeMix block.
        shift_state_for_tm_params = x_emb_prev # (1,1,D_MODEL)


        # --- Replicate TimeMix Weight Preparation ---
        # (B, T, C) where T=1
        B, T, C = x_for_tm_params.shape 
        
        # Token shift for parameter generation (eq.3)
        # x_shifted is shift_state_for_tm_params
        x_r_lerp = x_for_tm_params + (shift_state_for_tm_params - x_for_tm_params) * time_mix_block.mu_r
        x_k_lerp = x_for_tm_params + (shift_state_for_tm_params - x_for_tm_params) * time_mix_block.mu_k
        x_d_lerp = x_for_tm_params + (shift_state_for_tm_params - x_for_tm_params) * time_mix_block.mu_d
        x_a_lerp = x_for_tm_params + (shift_state_for_tm_params - x_for_tm_params) * time_mix_block.mu_a
        
        # Key precursor (k_t) (eq.5)
        k_vec = time_mix_block.W_k(x_k_lerp)    # (B,T,C) -> (1,1,D_MODEL)

        # Decay (w_t) (eq. 11, 12)
        d_lora_out = time_mix_block.decay_lora(x_d_lerp) 
        w_vec = torch.exp(-torch.exp(torch.tensor(-0.5, device=DEVICE, dtype=torch.float32)) * torch.sigmoid(d_lora_out.float())).type_as(x_for_tm_params)

        # In-context Learning Rate (a_t) (eq.4)
        a_vec = torch.sigmoid(time_mix_block.iclr_lora(x_a_lerp).float()).type_as(x_for_tm_params)

        # Removal Key (kappa_t) (eq.6)
        kappa_vec = k_vec * time_mix_block.removal_key_multiplier_xi 

        # --- Reshape for multi-head operations and select target head ---
        # All these vectors are (B, T, C) which is (1, 1, D_MODEL).
        # Reshape to (B, T, H, N) -> (1, 1, num_heads, head_size).
        w_head_all = w_vec.view(B, T, num_heads, head_size)
        a_head_all = a_vec.view(B, T, num_heads, head_size)
        kappa_head_all = kappa_vec.view(B, T, num_heads, head_size)

        # Normalized removal key (kappa_hat_t) (eq.15) per head
        kappa_hat_head_all = F.normalize(kappa_head_all, p=2, dim=-1) # Normalize over N dimension

        # Squeeze B and T dimensions as they are 1
        w_t_all_heads = w_head_all.squeeze(0).squeeze(0)             # (num_heads, head_size)
        a_t_all_heads = a_head_all.squeeze(0).squeeze(0)             # (num_heads, head_size)
        kappa_hat_t_all_heads = kappa_hat_head_all.squeeze(0).squeeze(0) # (num_heads, head_size)

    return w_t_all_heads.cpu().numpy(), \
           a_t_all_heads.cpu().numpy(), \
           kappa_hat_t_all_heads.cpu().numpy()


def calculate_actual_Gt_from_learned_params(w_t_head, a_t_head, kappa_hat_t_head):
    """
    Calculates the actual G_t for a single head using its learned parameters.
    The model's formula is G_t = diag(w_t) - kappa_hat_t^T @ (a_t . kappa_hat_t)
    """
    head_size = w_t_head.shape[0]
    
    diag_wt = np.diag(w_t_head) # (head_size, head_size)
    
    # kappa_hat_t_head is (head_size,)
    # a_t_head is (head_size,)
    term_a_kappa_hat = a_t_head * kappa_hat_t_head # element-wise, (head_size,)
    
    # Outer product: col_vec @ row_vec
    kappa_hat_col = kappa_hat_t_head.reshape(-1, 1) # (head_size, 1)
    term_a_kappa_hat_row = term_a_kappa_hat.reshape(1, -1) # (1, head_size)
    
    outer_product_term = kappa_hat_col @ term_a_kappa_hat_row # (head_size, head_size)
                                 
    Gt_actual = diag_wt - outer_product_term
    return Gt_actual

# --- Main Execution ---
def main():
    print(f"Using device: {DEVICE}")

    # --- Load Model and Config ---
    print("\n--- Loading Model and Configuration ---")
    vocab, vocab_size, target_substring = load_vocab_and_config(DATASET_FILE_CONFIG)
    if vocab is None:
        print("Exiting due to vocabulary loading failure.")
        return
    
    pad_token_id = vocab.get(PAD_TOKEN_CONFIG, 0) # Get PAD token ID

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

    # --- Experiment Parameters ---
    head_size = MODEL_HYPERPARAMETERS["HEAD_SIZE"]
    layer_to_inspect = 1 # Example: first layer
    head_to_inspect = 0  # Example: first head in that layer
    
    # Choose an input token and a previous token context
    # current_char_input = 'a'
    # prev_char_input = PAD_TOKEN_CONFIG # e.g., beginning of sequence
    
    # Let's try a few different inputs
    test_cases = [
        {'current_char': 'a', 'prev_char': PAD_TOKEN_CONFIG, 'desc': "Input 'a', prev PAD"},
        {'current_char': 'b', 'prev_char': 'a', 'desc': "Input 'b', prev 'a'"},
        {'current_char': 'c', 'prev_char': 'b', 'desc': "Input 'c', prev 'b'"},
        {'current_char': 'a', 'prev_char': 'c', 'desc': "Input 'a', prev 'c'"},
        {'current_char': 'b', 'prev_char': PAD_TOKEN_CONFIG, 'desc': "Input 'b', prev PAD"},
        {'current_char': 'c', 'prev_char': PAD_TOKEN_CONFIG, 'desc': "Input 'c', prev PAD"},
        {'current_char': 'a', 'prev_char': 'b', 'desc': "Input 'a', prev 'b'"},
        {'current_char': 'b', 'prev_char': 'c', 'desc': "Input 'b', prev 'c'"},
        {'current_char': 'c', 'prev_char': 'a', 'desc': "Input 'c', prev 'a'"}
    ]

    # Theoretical swap matrix (e.g., swapping index 0 and 1)
    idx_to_swap_1 = 0
    idx_to_swap_2 = 1 # Ensure distinct and within head_size
    if head_size <= 1: idx_to_swap_2 = 0 # Avoid error if head_size is 1
        
    theoretical_swap_c2_matrix = create_theoretical_swap_matrix_c2(head_size, idx_to_swap_1, idx_to_swap_2)
    print(f"\n--- Theoretical Swap Matrix (c=2, swaps idx {idx_to_swap_1} & {idx_to_swap_2}) ---")
    print(np.round(theoretical_swap_c2_matrix, decimals=3))
    print("-" * 50)

    for case in test_cases:
        current_char_input = case['current_char']
        prev_char_input = case['prev_char']
        description = case['desc']

        print(f"\n--- Analyzing Trained Model for: {description} ---")
        print(f"Layer: {layer_to_inspect}, Head: {head_to_inspect}")

        current_token_id = vocab.get(current_char_input)
        prev_token_id = vocab.get(prev_char_input)

        if current_token_id is None or prev_token_id is None:
            print(f"Error: Token(s) '{current_char_input}' or '{prev_char_input}' not in vocab. Skipping case.")
            continue

        # 1. Get learned parameters from the trained model for this input
        try:
            learned_w_all_h, learned_a_all_h, learned_kappa_hat_all_h = \
                get_learned_time_mix_params_for_input(model, current_token_id, prev_token_id, layer_to_inspect, vocab)
        except Exception as e:
            print(f"Error during parameter extraction for '{description}': {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Select parameters for the specific head
        learned_w_head = learned_w_all_h[head_to_inspect]
        learned_a_head = learned_a_all_h[head_to_inspect]
        learned_kappa_hat_head = learned_kappa_hat_all_h[head_to_inspect]

        # 2. Calculate the actual G_t matrix produced by this head for this input
        actual_Gt_matrix = calculate_actual_Gt_from_learned_params(
            learned_w_head, learned_a_head, learned_kappa_hat_head
        )
        
        print(f"\nActual G_t from Trained Model (Layer {layer_to_inspect}, Head {head_to_inspect}) for input '{current_char_input}' (prev '{prev_char_input}'):")
        print(np.round(actual_Gt_matrix, decimals=3))

        print("Eigenvalues of Actual G_t:")
        eigvals = np.linalg.eigvals(actual_Gt_matrix)
        print(np.round(eigvals, decimals=3))

        # 3. Compare with the theoretical swap matrix
        diff_matrix = actual_Gt_matrix - theoretical_swap_c2_matrix
        norm_of_difference = np.linalg.norm(diff_matrix)
        
        print(f"\nComparison with Theoretical Swap Matrix (swapping {idx_to_swap_1}&{idx_to_swap_2}):")
        print(f"  Norm of difference (Actual G_t - Theoretical Swap): {norm_of_difference:.4f}")
        
        if np.allclose(actual_Gt_matrix, theoretical_swap_c2_matrix, atol=1e-2): # Looser tolerance
            print("  The actual G_t is VERY CLOSE to the theoretical c=2 swap matrix for this input!")
            print("  This is surprising and suggests the model might have learned something akin to a swap for this specific context.")
        elif np.allclose(actual_Gt_matrix, 0.5 * np.eye(head_size) + 0.5 * theoretical_swap_c2_matrix, atol=1e-2):
            print("  The actual G_t is VERY CLOSE to the expected form for c=1 with Lemma 1's ideal inputs (0.5*I + 0.5*Swap_c2).")
            print("  This suggests the learned parameters (w,a,kappa_hat) are close to the idealized ones from Lemma 1.")
        else:
            print("  The actual G_t is NOT a close match to the theoretical swap matrix (or its c=1 scaled version).")
            print("  This is generally expected, as the model learns to optimize its primary task,")
            print("  not necessarily to form perfect idealized matrices for arbitrary inputs.")
        print("-" * 50)

    print("\nOverall Conclusion:")
    print("This script extracts the actual transition matrix parameters (w, a, kappa_hat) that your")
    print("TRAINED model computes for specific input tokens and contexts.")
    print("It then constructs the resulting G_t matrix for a chosen head and compares it to the")
    print("fixed, theoretical swap matrix from the RWKV-7 paper's Lemma 1 (which assumes c=2 and ideal parameters).")
    print("The differences highlight that while the architecture CAN represent a swap (theoretical proof),")
    print("the trained model's learned behavior for arbitrary inputs will generally be different,")
    print("as it's optimized for its specific regular language recognition task.")

if __name__ == "__main__":
    main()
