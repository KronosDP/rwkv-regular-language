import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

# --- Expected Imports from your project structure ---
# Assuming these files are in the same directory or PYTHONPATH is set up correctly
# Experiment-side imports
from experiment.rwkv_model import RWKV7_Model_Classifier
# Ensure this is the correct model class
# from config import (MODEL_HYPERPARAMETERS, DATASET_FILE_CONFIG, 
#                     MODEL_CHECKPOINT_PATH_CONFIG, PAD_TOKEN_CONFIG, MAX_LEN)
# Theoretical-side imports (classes from rwkv_constructor.py)
from theoretical.rwkv_constructor import (CopyMatrix, IdentityMatrix,
                                          SwapMatrix, WKVParams)

# --- Configuration ---
# These paths should point to your actual files


def find_model_file(filename, fallback_path):
    """Helper function to search for model files in current directory and subdirectories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for root, dirs, files in os.walk(current_dir):
        if filename in files:
            found_path = os.path.join(root, filename)
            print(f"Found {filename} at: {found_path}")
            return found_path
    
    print(f"Warning: '{filename}' not found in '{current_dir}' or subdirectories. Using fallback path: {fallback_path}")
    return fallback_path

# Search for model files
EXPERIMENTAL_MODEL_PATH = find_model_file(
    'rwkv7_fsm_experimental_model.pth', 
    '/experiment/rwkv7_fsm_experimental_model.pth'
)

CONCEPTUAL_MODEL_PATH = find_model_file(
    'rwkv7_fsm_conceptual_model.pth',
    '/theoretical/rwkv7_fsm_conceptual_model.pth'
)
# Path to the output of rwkv_constructor.py
DATASET_CONFIG_PATH = find_model_file(
    'regex_dataset.json',
    '/experiment/regex_dataset.json'
)
# Used for vocab and model params

# Model Hyperparameters (should match the trained experimental model)
# Load from your experiment/config.py or define here if static
# For simplicity, loading a subset. Ensure these match your experimental_model.py and config.py
# Example:
# D_MODEL = MODEL_HYPERPARAMETERS["D_MODEL"]
# N_LAYER = MODEL_HYPERPARAMETERS["N_LAYER"]
# HEAD_SIZE = MODEL_HYPERPARAMETERS["HEAD_SIZE"]
# FFN_HIDDEN_MULTIPLIER = MODEL_HYPERPARAMETERS["FFN_HIDDEN_MULTIPLIER"]
# LORA_DIM_W = MODEL_HYPERPARAMETERS["LORA_DIM_W"] 
# LORA_DIM_A = MODEL_HYPERPARAMETERS["LORA_DIM_A"]
# LORA_DIM_V = MODEL_HYPERPARAMETERS["LORA_DIM_V"]
# LORA_DIM_G = MODEL_HYPERPARAMETERS["LORA_DIM_G"]

# Placeholder - replace with actual loading or definitions from your config.py
MODEL_HYPERPARAMETERS_DEFAULT = {
    "D_MODEL": 8, "N_LAYER": 4, "HEAD_SIZE": 8, "FFN_HIDDEN_MULTIPLIER": 4,
    "LORA_DIM_W": 32, "LORA_DIM_A": 32, "LORA_DIM_V": 16, "LORA_DIM_G": 32
}
PAD_TOKEN_DEFAULT = '<pad>'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_RWKV_THEORETICAL = 2.0 # c value used in Lemma 4 for constructing G matrices

# Global variables to be loaded
VOCAB = {}
MODEL_MAX_LEN = 50
N_STATES_FSM = 0 # Will be loaded from conceptual model

# --- Utility Functions ---

def load_experimental_model_and_vocab():
    """Loads the experimental RWKV-7 model and vocabulary."""
    global VOCAB, MODEL_MAX_LEN, MODEL_HYPERPARAMETERS_DEFAULT
    
    model_hyperparams = MODEL_HYPERPARAMETERS_DEFAULT # Start with default
    pad_token = PAD_TOKEN_DEFAULT
    
    if os.path.exists(DATASET_CONFIG_PATH):
        try:
            with open(DATASET_CONFIG_PATH, 'r') as f:
                dataset_obj = json.load(f)
            VOCAB = dataset_obj['vocab']
            MODEL_MAX_LEN = dataset_obj.get('max_len', 50)
            # Potentially load experimental model's actual hyperparams if stored in dataset_obj
            # For now, we assume MODEL_HYPERPARAMETERS_DEFAULT is sufficient or user updates it.
            print(f"Vocabulary and max_len loaded from {DATASET_CONFIG_PATH}")
        except Exception as e:
            print(f"Warning: Could not load {DATASET_CONFIG_PATH}: {e}. Using default vocab if defined, or script may fail.")
            # Fallback VOCAB if needed, though it's best to ensure DATASET_CONFIG_PATH is correct
            if not VOCAB : VOCAB = {PAD_TOKEN_DEFAULT: 0, 'a':1, 'b':2, 'c':3} # Example fallback
    else:
        print(f"Warning: {DATASET_CONFIG_PATH} not found. Using default vocab/max_len.")
        if not VOCAB : VOCAB = {PAD_TOKEN_DEFAULT: 0, 'a':1, 'b':2, 'c':3}


    vocab_size = len(VOCAB)
    if vocab_size == 0:
        raise ValueError("Vocabulary is empty. Please check DATASET_CONFIG_PATH.")

    # --- Load Hyperparameters ---
    # This section should ideally load from your experiment/config.py
    # For now, we use the defaults and allow user to override
    # Example: from config import MODEL_HYPERPARAMETERS
    # model_hyperparams = MODEL_HYPERPARAMETERS
    
    print(f"Using Model Hyperparameters: {model_hyperparams}")

    model = RWKV7_Model_Classifier(
        d_model=model_hyperparams["D_MODEL"],
        n_layer=model_hyperparams["N_LAYER"],
        vocab_size=vocab_size,
        head_size=model_hyperparams["HEAD_SIZE"],
        ffn_hidden_multiplier=model_hyperparams["FFN_HIDDEN_MULTIPLIER"],
        lora_dim_w=model_hyperparams["LORA_DIM_W"],
        lora_dim_a=model_hyperparams["LORA_DIM_A"],
        lora_dim_v=model_hyperparams["LORA_DIM_V"],
        lora_dim_g=model_hyperparams["LORA_DIM_G"]
    ).to(DEVICE)

    if os.path.exists(EXPERIMENTAL_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(EXPERIMENTAL_MODEL_PATH, map_location=DEVICE))
            print(f"Experimental model weights loaded from {EXPERIMENTAL_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading experimental model weights: {e}")
            print("Proceeding with an uninitialized model for structural analysis if desired, but comparisons will be meaningless.")
    else:
        print(f"Warning: Experimental model path {EXPERIMENTAL_MODEL_PATH} not found. Using uninitialized model.")
    
    model.eval()
    return model, VOCAB

def load_conceptual_fsm_model():
    """Loads the conceptual FSM model generated by rwkv_constructor.py."""
    global N_STATES_FSM
    if not os.path.exists(CONCEPTUAL_MODEL_PATH):
        raise FileNotFoundError(f"Conceptual model {CONCEPTUAL_MODEL_PATH} not found. Run rwkv_constructor.py first.")
    try:
        # Try torch.load first
        conceptual_model = torch.load(CONCEPTUAL_MODEL_PATH, map_location=DEVICE)
        print(f"Conceptual FSM model loaded from {CONCEPTUAL_MODEL_PATH} using torch.load().")
    except Exception:
        print(f"torch.load() failed for {CONCEPTUAL_MODEL_PATH}. Attempting pickle.")
        with open(CONCEPTUAL_MODEL_PATH, 'rb') as f:
            conceptual_model = pickle.load(f)
        print(f"Conceptual FSM model loaded from {CONCEPTUAL_MODEL_PATH} using pickle.")
    
    N_STATES_FSM = conceptual_model.get('dfa_representation', {}).get('n_states', 0)
    if N_STATES_FSM == 0:
        print("Warning: Loaded conceptual FSM model has 0 states.")
    return conceptual_model

def get_theoretical_elementary_op_and_params(conceptual_model, symbol, op_index):
    """
    Retrieves a specific theoretical elementary operation matrix G and its WKV parameters
    from the loaded conceptual model.
    Returns (None, None) if any error occurs.
    """
    if N_STATES_FSM == 0:
        # print("Error: FSM has 0 states. Cannot get elementary operation.") # Already printed by load_conceptual_fsm_model
        return None, None

    factorized_M_sigma = conceptual_model.get('factorized_M_sigma', {})
    if symbol not in factorized_M_sigma:
        print(f"Error: Symbol '{symbol}' not found in factorized_M_sigma of conceptual model.")
        return None, None
    
    ops_for_symbol_str = factorized_M_sigma[symbol]
    if not isinstance(ops_for_symbol_str, list) or op_index >= len(ops_for_symbol_str):
        print(f"Error: op_index {op_index} out of bounds for symbol '{symbol}' (max: {len(ops_for_symbol_str)-1 if isinstance(ops_for_symbol_str, list) else 'N/A'}). Ops: {ops_for_symbol_str}")
        return None, None

    op_str = ops_for_symbol_str[op_index]
    
    elem_op_obj = None
    try:
        if "IdentityMatrix" in op_str:
            elem_op_obj = IdentityMatrix(N_STATES_FSM)
        elif "SwapMatrix" in op_str:
            parts = op_str.replace("SwapMatrix(size=", "").replace(")", "").split(", i=")
            i_val, j_val = map(int, parts[1].split(", j="))
            elem_op_obj = SwapMatrix(N_STATES_FSM, i_val, j_val)
        elif "CopyMatrix" in op_str:
            parts = op_str.replace("CopyMatrix(size=", "").replace(")", "").split(", col_to_replace=")
            replace_val, copy_from_val = map(int, parts[1].split(", col_to_copy_from="))
            elem_op_obj = CopyMatrix(N_STATES_FSM, replace_val, copy_from_val)
    except Exception as e:
        print(f"Error parsing elementary operation string '{op_str}': {e}")
        return None, None

    if elem_op_obj is None:
        print(f"Error: Could not parse elementary operation string: {op_str}")
        return None, None

    try:
        wkv_params_theoretical = WKVParams.get_wkv_params_for_elementary_matrix(elem_op_obj, N_STATES_FSM, c_rwkv=C_RWKV_THEORETICAL)
        if not isinstance(wkv_params_theoretical, dict): # Should not happen based on WKVParams structure
             print(f"Error: WKVParams.get_wkv_params_for_elementary_matrix did not return a dictionary for {op_str}")
             return elem_op_obj.matrix, None # Return matrix if object was created, but params failed
    except Exception as e:
        print(f"Error getting WKV params for elementary op {op_str}: {e}")
        return elem_op_obj.matrix, None # Return matrix if object was created, but params failed
        
    G_theoretical_matrix = elem_op_obj.matrix 
    
    return G_theoretical_matrix, wkv_params_theoretical


def get_learned_wkv_components(experimental_model, input_token_ids, layer_id_to_inspect):
    """
    Performs a partial forward pass on the experimental model to get learned
    w_t, a_t, kappa_hat_t, and the full WKV state for specific input_token_ids and layer.
    Returns these parameters for ALL heads in the specified layer at the LAST time step of input_token_ids.
    """
    if layer_id_to_inspect >= experimental_model.n_layer:
        print(f"Error: layer_id_to_inspect ({layer_id_to_inspect}) is out of bounds (max: {experimental_model.n_layer-1}).")
        return None
    
    input_tensor = torch.tensor([input_token_ids], device=DEVICE, dtype=torch.long)
    B, T_seq = input_tensor.shape
    
    if T_seq == 0:
        print("Error: Input token sequence is empty.")
        return None
        
    extracted_params = {}

    with torch.no_grad():
        x_emb = experimental_model.embedding(input_tensor) 
        initial_shift_state_for_vpc = torch.zeros(B, 1, experimental_model.d_model, device=DEVICE, dtype=x_emb.dtype)
        
        x_emb_shifted_for_vpc_list = []
        current_shift_for_vpc = initial_shift_state_for_vpc
        for t_step_vpc in range(T_seq): # Corrected loop variable name
            x_emb_shifted_for_vpc_list.append(current_shift_for_vpc)
            current_shift_for_vpc = x_emb[:, t_step_vpc:t_step_vpc+1, :] 
        x_emb_shifted_for_vpc = torch.cat(x_emb_shifted_for_vpc_list, dim=1) 

        x_v_lerp_for_vpc = x_emb + (x_emb_shifted_for_vpc - x_emb) * experimental_model.mu_v_for_v_prime_c
        v_prime_c = experimental_model.W_v_for_v_prime_c(x_v_lerp_for_vpc) 

        current_x = experimental_model.ln_pre_blocks(x_emb)
        
        tm_shift_state_prev = torch.zeros(B, 1, experimental_model.d_model, device=DEVICE, dtype=current_x.dtype)
        tm_wkv_state_prev = torch.zeros(B, experimental_model.num_heads, experimental_model.head_size, experimental_model.head_size, device=DEVICE, dtype=current_x.dtype)
        cm_shift_state_prev = torch.zeros(B, 1, experimental_model.d_model, device=DEVICE, dtype=current_x.dtype)

        # For storing parameters for the target layer across time
        learned_w_vec_target_layer_all_t = []
        learned_a_vec_target_layer_all_t = []
        learned_kappa_hat_head_target_layer_all_t = []

        for i in range(experimental_model.n_layer):
            block = experimental_model.blocks[i]
            time_mix_block = block.tm
            
            # --- Time Mixing Input for the current block ---
            tm_input_norm = block.ln_tm_in(current_x) # current_x is output from previous block/embedding LN

            # --- Iterate through time to extract WKV components if it's the target layer ---
            # This re-computation is to get intermediate w_t, a_t, kappa_hat_t at each time step
            # for the layer_id_to_inspect.
            if i == layer_id_to_inspect:
                # We need the shift state *for this specific TimeMix block* as it evolves.
                # The `tm_shift_state_prev` is the state *entering* the block.
                # Within the block, for each t_step, x_shifted_tm is tm_input_norm's previous time step's value.
                
                # Token shift *within* the TimeMix block for its input tm_input_norm
                # tm_shift_state_for_block_internal is effectively tm_input_norm[:, t-1, :]
                initial_tm_internal_shift = torch.zeros(B,1,experimental_model.d_model, device=DEVICE, dtype=tm_input_norm.dtype)
                if tm_shift_state_prev is not None : # Use state from previous layer/block
                     initial_tm_internal_shift = tm_shift_state_prev


                tm_input_shifted_internally_list = []
                current_internal_shift = initial_tm_internal_shift
                for t_vpc_tm in range(T_seq): # loop var name
                    tm_input_shifted_internally_list.append(current_internal_shift)
                    current_internal_shift = tm_input_norm[:, t_vpc_tm:t_vpc_tm+1, :]
                tm_input_shifted_internally = torch.cat(tm_input_shifted_internally_list, dim=1)


                for t_step in range(T_seq):
                    tm_input_t = tm_input_norm[:, t_step:t_step+1, :]
                    # x_shifted_tm for THIS timestep t_step is tm_input_norm[:, t_step-1, :]
                    # which is correctly tm_input_shifted_internally[:, t_step:t_step+1, :]
                    x_shifted_tm_t = tm_input_shifted_internally[:, t_step:t_step+1, :]
                    
                    x_k_lerp_tm = tm_input_t + (x_shifted_tm_t - tm_input_t) * time_mix_block.mu_k
                    x_d_lerp_tm = tm_input_t + (x_shifted_tm_t - tm_input_t) * time_mix_block.mu_d
                    x_a_lerp_tm = tm_input_t + (x_shifted_tm_t - tm_input_t) * time_mix_block.mu_a
                    
                    k_vec_tm = time_mix_block.W_k(x_k_lerp_tm)
                    d_lora_out_tm = time_mix_block.decay_lora(x_d_lerp_tm)
                    w_vec_tm = torch.exp(-torch.exp(torch.tensor(-0.5, device=DEVICE, dtype=torch.float32)) * torch.sigmoid(d_lora_out_tm.float())).type_as(tm_input_t)
                    a_vec_tm = torch.sigmoid(time_mix_block.iclr_lora(x_a_lerp_tm).float()).type_as(tm_input_t)
                    kappa_vec_tm = k_vec_tm * time_mix_block.removal_key_multiplier_xi
                    
                    kappa_head_tm = kappa_vec_tm.view(B, 1, time_mix_block.num_heads, time_mix_block.head_size)
                    kappa_hat_head_tm = F.normalize(kappa_head_tm, p=2, dim=-1)

                    learned_w_vec_target_layer_all_t.append(w_vec_tm.squeeze(1))
                    learned_a_vec_target_layer_all_t.append(a_vec_tm.squeeze(1))
                    learned_kappa_hat_head_target_layer_all_t.append(kappa_hat_head_tm.squeeze(1))
            
            # --- Actual block forward pass (for the whole input_tensor sequence for this block) ---
            _dx_tm, _next_tm_ss, _next_tm_ws, cm_shift_state_for_current_block_output = experimental_model.blocks[i](
                current_x, v_prime_c, # v_prime_c is (B, T_seq, C)
                tm_shift_state_prev, tm_wkv_state_prev, cm_shift_state_prev
            )
            
            if i == layer_id_to_inspect:
                extracted_params['wkv_state_final_experimental'] = _next_tm_ws.detach().cpu().numpy()
            
            current_x = current_x + _dx_tm 
            
            cm_input = experimental_model.blocks[i].ln_cm_in(current_x)
            # The cm_shift_state_for_current_block_output from the block's return is the *next* cm_shift_state
            # The cm_shift_state_prev for *this* block's CM was cm_shift_state_prev
            _dx_cm, _next_cm_ss = experimental_model.blocks[i].cm(cm_input, cm_shift_state_prev)
            current_x = current_x + _dx_cm
            
            tm_shift_state_prev = _next_tm_ss
            tm_wkv_state_prev = _next_tm_ws
            cm_shift_state_prev = _next_cm_ss


    if learned_w_vec_target_layer_all_t: # Check if lists were populated
        extracted_params['w_vec_learned'] = torch.stack(learned_w_vec_target_layer_all_t, dim=1).detach().cpu().numpy()[:, -1, :]
        extracted_params['a_vec_learned'] = torch.stack(learned_a_vec_target_layer_all_t, dim=1).detach().cpu().numpy()[:, -1, :]
        extracted_params['kappa_hat_learned'] = torch.stack(learned_kappa_hat_head_target_layer_all_t, dim=1).detach().cpu().numpy()[:, -1, :, :]
    else: # If layer_id_to_inspect was not met or T_seq=0 led to empty lists
        if i == layer_id_to_inspect : # If it was the target layer but lists are empty (e.g. T_seq=0)
             print(f"Warning: Target layer {layer_id_to_inspect} was processed, but no WKV components (w,a,k_hat) extracted. Input sequence length T_seq might be 0.")
        # extracted_params will not have 'w_vec_learned', 'a_vec_learned', 'kappa_hat_learned'
        # It might still have 'wkv_state_final_experimental' if T_seq > 0

    return extracted_params

def reconstruct_Gt_experimental(w_vec_learned_head, a_vec_learned_head, kappa_hat_learned_head, c_rwkv_experimental=1.0):
    """
    Reconstructs the G_t matrix for a single head from learned parameters.
    Assumes w_vec and a_vec are for one head (length N=head_size).
    kappa_hat_learned_head is (N).
    """
    N = kappa_hat_learned_head.shape[-1]
    diag_wt = np.diag(w_vec_learned_head) 
    
    kappa_hat_T_col = kappa_hat_learned_head.reshape(N, 1)
    a_dot_kappa_hat_row = (a_vec_learned_head * kappa_hat_learned_head).reshape(1, N)
    outer_product_term = kappa_hat_T_col @ a_dot_kappa_hat_row
    
    Gt_experimental = diag_wt - c_rwkv_experimental * outer_product_term
    return Gt_experimental

def compare_matrices(M1, M2, title1="Matrix 1", title2="Matrix 2"):
    """Compares two matrices using norm difference and cosine similarity, and plots them."""
    if M1 is None or M2 is None:
        print("One of the matrices is None, cannot compare.")
        return
    if M1.shape != M2.shape:
        print(f"Matrix shapes differ: {M1.shape} vs {M2.shape}. Cannot compare directly.")
        return

    diff_norm = np.linalg.norm(M1 - M2)
    
    m1_flat = M1.flatten()
    m2_flat = M2.flatten()
    if np.linalg.norm(m1_flat) == 0 or np.linalg.norm(m2_flat) == 0:
        cosine_sim = 0.0 if np.array_equal(m1_flat,m2_flat) else -1.0 
        if not (np.linalg.norm(m1_flat) == 0 and np.linalg.norm(m2_flat) == 0 and np.array_equal(m1_flat, m2_flat)):
             print("Warning: Zero vector encountered in cosine similarity calculation.")
    else:
        cosine_sim = np.dot(m1_flat, m2_flat) / (np.linalg.norm(m1_flat) * np.linalg.norm(m2_flat))
    
    print(f"\n--- Matrix Comparison ---")
    print(f"Norm of difference ({title1} - {title2}): {diff_norm:.4f}")
    print(f"Cosine similarity: {cosine_sim:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Determine robust min/max for color scale, avoiding issues if one matrix is all zeros
    all_vals = np.concatenate((M1.flatten(), M2.flatten(), (M1-M2).flatten()))
    if len(all_vals) > 0 and not np.all(all_vals == 0):
        v_min = np.percentile(all_vals, 1)
        v_max = np.percentile(all_vals, 99)
        if v_min == v_max: # Handle case where all values are almost the same
            v_min -= 0.1
            v_max += 0.1
        if v_min == 0 and v_max == 0: # Handle all zeros
            v_min = -0.1
            v_max = 0.1

    else: # All matrices are zero or empty
        v_min, v_max = -0.1, 0.1
        
    center_val = 0

    sns.heatmap(M1, annot=True, fmt='.2f', cmap='RdBu_r', center=center_val, ax=axes[0])
    axes[0].set_title(title1)
    
    sns.heatmap(M2, annot=True, fmt='.2f', cmap='RdBu_r', center=center_val, ax=axes[1])
    axes[1].set_title(title2)
    
    sns.heatmap(M1 - M2, annot=True, fmt='.2f', cmap='RdBu_r', center=center_val, ax=axes[2])
    axes[2].set_title(f"Difference ({title1} - {title2})")
    
    plt.tight_layout()
    plt.show()

def simulate_theoretical_fsm_state_evolution(conceptual_model, input_string_chars):
    """Simulates the FSM from the conceptual model and returns the state sequence."""
    if not conceptual_model or 'dfa_representation' not in conceptual_model:
        print("Error: Conceptual model is invalid or missing 'dfa_representation'.")
        return []
        
    dfa_rep = conceptual_model['dfa_representation']
    n_states = dfa_rep.get('n_states')
    if n_states is None or n_states == 0: 
        print("Error: FSM n_states is 0 or not defined in conceptual model.")
        return []

    state_to_idx = dfa_rep.get('state_to_idx')
    initial_state_val = conceptual_model.get('parsed_fsm', {}).get('initial_state')
    M_sigma_matrices_list = dfa_rep.get('M_sigma_matrices')

    if not all([state_to_idx, initial_state_val is not None, M_sigma_matrices_list]):
        print("Error: Missing essential DFA components (state_to_idx, initial_state, M_sigma_matrices) in conceptual model.")
        return []
        
    if initial_state_val not in state_to_idx: 
        print(f"Error: Initial state {initial_state_val} not in state_to_idx map.")
        return []
        
    current_state_idx = state_to_idx[initial_state_val]
    
    M_sigma_matrices = {
        symbol: np.array(matrix) for symbol, matrix in M_sigma_matrices_list.items()
    }
    
    theoretical_states_one_hot = []
    initial_state_vec = np.zeros(n_states, dtype=int)
    initial_state_vec[current_state_idx] = 1
    theoretical_states_one_hot.append(initial_state_vec.copy())

    current_state_vector = initial_state_vec
    for symbol in input_string_chars:
        if symbol not in M_sigma_matrices:
            print(f"Symbol '{symbol}' not in FSM alphabet for theoretical simulation.")
            theoretical_states_one_hot.append(current_state_vector.copy()) 
            continue 
        
        M = M_sigma_matrices[symbol]
        if M.shape[1] != len(current_state_vector):
            print(f"Shape mismatch: M_sigma for '{symbol}' ({M.shape}) and current_state_vector ({current_state_vector.shape})")
            theoretical_states_one_hot.append(current_state_vector.copy())
            continue

        next_state_vector = np.dot(M, current_state_vector)
        
        if np.sum(next_state_vector) != 1: 
            print(f"Warning: Next state vector for symbol '{symbol}' does not sum to 1: {next_state_vector}. Using previous state.")
            theoretical_states_one_hot.append(current_state_vector.copy())
        else:
            current_state_vector = next_state_vector
            theoretical_states_one_hot.append(current_state_vector.copy())
            
    return theoretical_states_one_hot 

# --- Main Analysis Functions ---

def analyze_parameter_level_comparison(experimental_model, conceptual_fsm_model,
                                     input_char_sequence_str,
                                     target_symbol_in_fsm, 
                                     target_elementary_op_index_for_symbol,
                                     layer_to_inspect, head_to_inspect):
    """
    Compares a theoretical elementary operation G_target with the G_experimental
    learned by the model for a given input context.
    """
    print(f"\n--- Parameter-Level Comparison ---")
    print(f"Input sequence for WKV components: '{input_char_sequence_str}' (analysis at last token)")
    print(f"Target FSM symbol for theoretical G: '{target_symbol_in_fsm}', op_index: {target_elementary_op_index_for_symbol}")
    print(f"Inspecting Layer {layer_to_inspect}, Head {head_to_inspect} of experimental model.")

    if N_STATES_FSM == 0: # N_STATES_FSM is global, set by load_conceptual_fsm_model
        print("FSM has 0 states. Aborting parameter analysis.")
        return

    # 1. Get Theoretical G and WKV parameters
    G_theoretical, wkv_params_theoretical = get_theoretical_elementary_op_and_params(
        conceptual_fsm_model, target_symbol_in_fsm, target_elementary_op_index_for_symbol
    )
    
    if G_theoretical is None or wkv_params_theoretical is None:
        print(f"Error: Failed to get theoretical G matrix or WKV parameters for symbol '{target_symbol_in_fsm}', op_index {target_elementary_op_index_for_symbol}.")
        if G_theoretical is not None and wkv_params_theoretical is None:
            print("  Details: G_theoretical matrix was retrieved, but wkv_params_theoretical is None. This indicates an issue in WKVParams.get_wkv_params_for_elementary_matrix or its call chain.")
        elif G_theoretical is None and wkv_params_theoretical is not None:
             print("  Details: G_theoretical is None, but wkv_params_theoretical was retrieved. This is unexpected from get_theoretical_elementary_op_and_params.")
        return
    
    # Now it's safe to access wkv_params_theoretical keys
    print(f"Theoretical elementary operation: {wkv_params_theoretical.get('comment', 'N/A')}")

    # 2. Get Learned WKV components
    input_token_ids = [VOCAB.get(char, VOCAB.get(PAD_TOKEN_DEFAULT,0)) for char in input_char_sequence_str]
    if not input_token_ids:
        print("Error: Input character sequence is empty or results in empty token ID list.")
        return

    learned_components = get_learned_wkv_components(experimental_model, input_token_ids, layer_to_inspect)
    
    if learned_components is None or \
       not learned_components.get('w_vec_learned') is not None or \
       not learned_components.get('a_vec_learned') is not None or \
       not learned_components.get('kappa_hat_learned') is not None:
        print("Could not extract all required learned WKV components (w, a, kappa_hat).")
        return

    # These are now for the last time step, shape (B,C) or (B,H,N)
    w_learned_batch = learned_components['w_vec_learned'] 
    a_learned_batch = learned_components['a_vec_learned'] 
    kappa_hat_learned_batch = learned_components['kappa_hat_learned']

    # Assuming Batch size B=1 for these component extractions
    w_learned_all_heads = w_learned_batch[0] 
    a_learned_all_heads = a_learned_batch[0] 
    kappa_hat_learned_all_heads = kappa_hat_learned_batch[0] 

    num_heads = experimental_model.num_heads
    head_size_exp = experimental_model.head_size

    if head_to_inspect >= num_heads:
        print(f"Error: head_to_inspect ({head_to_inspect}) is out of bounds for experimental model (num_heads: {num_heads}).")
        return

    w_learned_this_head = w_learned_all_heads[head_to_inspect*head_size_exp : (head_to_inspect+1)*head_size_exp]
    a_learned_this_head = a_learned_all_heads[head_to_inspect*head_size_exp : (head_to_inspect+1)*head_size_exp]
    kappa_hat_learned_this_head = kappa_hat_learned_all_heads[head_to_inspect, :] 

    # 3. Reconstruct G_experimental
    G_theoretical_for_comparison = G_theoretical # Store original before potential resize
    if head_size_exp != N_STATES_FSM:
        print(f"Warning: Experimental model head size ({head_size_exp}) != FSM n_states ({N_STATES_FSM}).")
        print("Direct G matrix comparison might be misleading. Attempting to use a resized/re-derived theoretical G.")
        
        temp_op_str = wkv_params_theoretical.get('comment', "IdentityMatrix") # Default if comment missing
        temp_elem_op_resized = None
        if "IdentityMatrix" in temp_op_str: temp_elem_op_resized = IdentityMatrix(head_size_exp)
        elif "SwapMatrix" in temp_op_str and head_size_exp >=2:
             try:
                indices_str = temp_op_str.split('(')[1].split(')')[0]
                # Parsing "size=X, i=Y, j=Z" or "X, Y" for Swap
                parsed_indices = [int(p.split('=')[-1].strip()) for p in indices_str.split(',') if 'i=' in p or 'j=' in p]
                if len(parsed_indices) == 2 :
                    i_swap, j_swap = parsed_indices[0], parsed_indices[1]
                    if i_swap < head_size_exp and j_swap < head_size_exp:
                        temp_elem_op_resized = SwapMatrix(head_size_exp, i_swap, j_swap)
                    else: temp_elem_op_resized = IdentityMatrix(head_size_exp)
                else: temp_elem_op_resized = IdentityMatrix(head_size_exp)
             except Exception as e_parse: 
                print(f"  Could not parse swap indices from '{temp_op_str}': {e_parse}. Using Identity.")
                temp_elem_op_resized = IdentityMatrix(head_size_exp)
        elif "CopyMatrix" in temp_op_str and head_size_exp >=1:
            try:
                indices_str = temp_op_str.split('(')[1].split(')')[0]
                parsed_indices = [int(p.split('=')[-1].strip()) for p in indices_str.split(',') if 'col_to_replace=' in p or 'col_to_copy_from=' in p]
                if len(parsed_indices) == 2:
                    replace_val, copy_from_val = parsed_indices[0], parsed_indices[1]
                    if replace_val < head_size_exp and copy_from_val < head_size_exp:
                         temp_elem_op_resized = CopyMatrix(head_size_exp, replace_val, copy_from_val)
                    else: temp_elem_op_resized = IdentityMatrix(head_size_exp)
                else: temp_elem_op_resized = IdentityMatrix(head_size_exp)
            except Exception as e_parse:
                print(f"  Could not parse copy indices from '{temp_op_str}': {e_parse}. Using Identity.")
                temp_elem_op_resized = IdentityMatrix(head_size_exp)
        else: temp_elem_op_resized = IdentityMatrix(head_size_exp)
        
        # Create a temporary conceptual model structure for get_theoretical_elementary_op_and_params
        # to get a G_theoretical of the experimental head size.
        # We are only interested in the matrix from this, not its wkv_params.
        G_theoretical_resized_matrix, _ = get_theoretical_elementary_op_and_params(
            {'factorized_M_sigma': {target_symbol_in_fsm: [str(temp_elem_op_resized)]}, 
             'dfa_representation': {'n_states': head_size_exp}, # Use exp head size as n_states
             'parsed_fsm': {'initial_state': 0} # Dummy
             },
            target_symbol_in_fsm, 0 
        )
        if G_theoretical_resized_matrix is not None:
            G_theoretical_for_comparison = G_theoretical_resized_matrix
            print(f"  Using a re-derived G_theoretical of shape {G_theoretical_for_comparison.shape} for comparison based on '{str(temp_elem_op_resized)}'.")
        else:
            print(f"  Could not re-derive G_theoretical for comparison. Original shape: {G_theoretical.shape}, Experimental head shape: ({head_size_exp},{head_size_exp}).")
            # Proceed with original G_theoretical, comparison will likely fail shape check in compare_matrices

    G_experimental = reconstruct_Gt_experimental(w_learned_this_head, a_learned_this_head, kappa_hat_learned_this_head, c_rwkv_experimental=1.0)
    
    compare_matrices(G_theoretical_for_comparison, G_experimental, title1="Theoretical G", title2="Experimental G")


def analyze_state_evolution_comparison(experimental_model, conceptual_fsm_model, input_string_chars, layer_to_analyze_exp_state):
    """
    Compares the theoretical FSM state evolution with the experimental model's WKV state.
    """
    print(f"\n--- State Evolution Comparison for input: '{''.join(input_string_chars)}' ---")
    
    theoretical_states = simulate_theoretical_fsm_state_evolution(conceptual_fsm_model, input_string_chars)
    if not theoretical_states:
        print("Could not simulate theoretical FSM states.")
        return
    
    print(f"Theoretical FSM state sequence (len {len(theoretical_states)}):")
    # for i, state_vec in enumerate(theoretical_states):
    #     print(f"  t={i}: {state_vec} (state_idx: {np.argmax(state_vec)})")

    input_token_ids = [VOCAB.get(char, VOCAB.get(PAD_TOKEN_DEFAULT,0)) for char in input_string_chars]
    if not input_token_ids:
        print("Error: Input string resulted in empty token ID list for state evolution.")
        return

    learned_components = get_learned_wkv_components(experimental_model, input_token_ids, layer_to_analyze_exp_state)

    if learned_components is None or 'wkv_state_final_experimental' not in learned_components:
        print(f"Could not get experimental WKV state for layer {layer_to_analyze_exp_state}.")
        return
        
    experimental_wkv_state_final_batch = learned_components['wkv_state_final_experimental'] 
    experimental_wkv_state_final = experimental_wkv_state_final_batch[0] 
    
    print(f"Shape of final experimental WKV state (Layer {layer_to_analyze_exp_state}): {experimental_wkv_state_final.shape} (H, N, N)")

    final_theoretical_state_vec = theoretical_states[-1] 
    print(f"Final theoretical FSM state vector: {final_theoretical_state_vec}")

    n_exp_heads, n_exp_head_size, _ = experimental_wkv_state_final.shape
    
    # N_STATES_FSM is global
    if n_exp_head_size == N_STATES_FSM and N_STATES_FSM > 0 :
        print(f"Experimental head_size ({n_exp_head_size}) matches FSM n_states ({N_STATES_FSM}).")
        
        fig, axes = plt.subplots(1, min(n_exp_heads, 4) + 1, figsize=(5*(min(n_exp_heads,4)+1), 4))
        
        sns.heatmap(final_theoretical_state_vec.reshape(-1,1), annot=True, fmt='.2f', cmap='viridis', ax=axes[0], cbar=False)
        axes[0].set_title(f"Theoretical State\n(argmax: {np.argmax(final_theoretical_state_vec)})")
        axes[0].set_xlabel("State Vector")
        axes[0].set_ylabel("Dimension")

        for head_idx_plot in range(min(n_exp_heads, 4)): 
            first_row_wkv_head = experimental_wkv_state_final[head_idx_plot, 0, :] 
            
            sns.heatmap(first_row_wkv_head.reshape(-1,1), annot=True, fmt='.2f', cmap='viridis', ax=axes[head_idx_plot+1])
            axes[head_idx_plot+1].set_title(f"Exp. WKV Head {head_idx_plot}\n(First Row)")
            axes[head_idx_plot+1].set_xlabel("State Vector")
            axes[head_idx_plot+1].set_ylabel("Dimension")

            if np.linalg.norm(first_row_wkv_head) > 1e-6 and np.linalg.norm(final_theoretical_state_vec) > 1e-6:
                cos_sim = np.dot(first_row_wkv_head, final_theoretical_state_vec) / \
                          (np.linalg.norm(first_row_wkv_head) * np.linalg.norm(final_theoretical_state_vec))
                print(f"  Head {head_idx_plot} (1st row) vs Theoretical State: Cosine Sim = {cos_sim:.3f}")
            else:
                print(f"  Head {head_idx_plot} (1st row) vs Theoretical State: Cosine Sim = N/A (zero vector)")

        plt.suptitle(f"State Comparison for input '{''.join(input_string_chars)}', Exp Layer {layer_to_analyze_exp_state}")
        plt.show()
    elif N_STATES_FSM == 0:
        print("Skipping state evolution plot as FSM has 0 states.")
    else: # Mismatch
        print(f"Experimental head_size ({n_exp_head_size}) != FSM n_states ({N_STATES_FSM}). Direct state vector comparison is complex.")


# --- Main Execution ---
if __name__ == "__main__":
    print("RWKV-7 Theoretical vs. Experimental Analysis Toolkit")
    print(f"Using device: {DEVICE}")

    try:
        experimental_model, _vocab = load_experimental_model_and_vocab()
        VOCAB = _vocab 
        conceptual_fsm_model = load_conceptual_fsm_model()
    except Exception as e:
        print(f"Error during setup: {e}")
        exit()

    if N_STATES_FSM == 0: # Global N_STATES_FSM is set by load_conceptual_fsm_model
        print("Conceptual FSM model has 0 states. Many analyses will not be meaningful.")
    
    example_param_input_seq_for_a_after_b = "ba"
    fsm_symbol_to_analyze = 'a'
    elementary_op_idx = 0 
    exp_layer_to_inspect_params = 0 
    exp_head_to_inspect_params = 0  

    # Run parameter comparison regardless of N_STATES_FSM initially, 
    # as analyze_parameter_level_comparison has its own N_STATES_FSM check
    analyze_parameter_level_comparison(
        experimental_model, conceptual_fsm_model,
        input_char_sequence_str=example_param_input_seq_for_a_after_b,
        target_symbol_in_fsm=fsm_symbol_to_analyze,
        target_elementary_op_index_for_symbol=elementary_op_idx,
        layer_to_inspect=exp_layer_to_inspect_params,
        head_to_inspect=exp_head_to_inspect_params
    )

    example_state_input_string = "ababa" 
    exp_layer_to_inspect_state = experimental_model.n_layer -1 if experimental_model.n_layer > 0 else 0

    # Run state evolution comparison if N_STATES_FSM > 0
    if N_STATES_FSM > 0 :
        analyze_state_evolution_comparison(
            experimental_model, conceptual_fsm_model,
            list(example_state_input_string),
            layer_to_analyze_exp_state=exp_layer_to_inspect_state
        )
    else:
        print("\nSkipping state evolution comparison as loaded FSM has 0 states.")

    print("\nAnalysis toolkit finished.")

