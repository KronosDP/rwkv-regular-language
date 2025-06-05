import pickle

import numpy as np

# Attempt to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def load_conceptual_model(pth_file_path):
    """Loads the conceptual model from a .pth file."""
    loaded_config = None
    if TORCH_AVAILABLE:
        try:
            # Try torch.load first, as it's the primary way for .pth
            loaded_config = torch.load(pth_file_path)
            print(f"Successfully loaded model config from {pth_file_path} using torch.load().")
        except Exception as e_torch:
            print(f"torch.load() failed: {e_torch}. Attempting to load with pickle...")
            try:
                with open(pth_file_path, 'rb') as f:
                    loaded_config = pickle.load(f)
                print(f"Successfully loaded model config from {pth_file_path} using pickle.")
            except Exception as e_pickle:
                print(f"Pickle load also failed: {e_pickle}.")
                raise IOError(f"Could not load model config from {pth_file_path} using torch or pickle.")
    else: # Torch not available, must use pickle
        try:
            with open(pth_file_path, 'rb') as f:
                loaded_config = pickle.load(f)
            print(f"Successfully loaded model config from {pth_file_path} using pickle (PyTorch not available).")
        except Exception as e_pickle:
            print(f"Pickle load failed: {e_pickle}.")
            raise IOError(f"Could not load model config from {pth_file_path} (PyTorch not available).")
    return loaded_config

def simulate_fsm_from_config(config, input_string):
    """
    Simulates the FSM based on the DFA representation in the config.
    This does NOT run an RWKV-7 model, but simulates the DFA logic
    that the theoretical RWKV-7 construction aims to replicate.

    Args:
        config (dict): The loaded rwkv7_config.
        input_string (str or list): A string of input symbols (e.g., "aba")
                                     or a list of symbols (e.g., ['a','b','a']).

    Returns:
        bool: True if the string is accepted, False otherwise.
        list: Sequence of states traversed.
    """
    dfa_rep = config['dfa_representation']
    n_states = dfa_rep['n_states']
    
    if n_states == 0:
        print("Warning: FSM has no states defined. Cannot simulate.")
        # An FSM with no states might accept an empty string if initial is also accept and no alphabet,
        # or reject everything. For simplicity, reject non-empty strings.
        return (True if not input_string and not dfa_rep['alphabet'] and 
                config['parsed_fsm']['initial_state'] in config['parsed_fsm']['accept_states'] 
                else False), []


    state_to_idx = dfa_rep['state_to_idx']
    idx_to_state = {v: k for k, v in state_to_idx.items()}
    
    initial_state_val = config['parsed_fsm']['initial_state']
    if initial_state_val not in state_to_idx:
        print(f"Error: Initial state {initial_state_val} not in state map. Cannot simulate.")
        return False, []
        
    current_state_idx = state_to_idx[initial_state_val]
    
    # Convert alpha vector (one-hot) to current state vector for simulation
    # alpha_vec = np.array(dfa_rep['alpha_vec (initial_state_one_hot)']) # current state as a vector
    
    M_sigma_matrices = {
        symbol: np.array(matrix) for symbol, matrix in dfa_rep['M_sigma_matrices'].items()
    }
    omega_vec = np.array(dfa_rep['omega_vec (accept_states_multi_hot)'])
    alphabet = dfa_rep['alphabet']

    traversed_states_actual = [idx_to_state[current_state_idx]] # Store actual state values

    print(f"Simulating FSM from config: {config['fsm_source_file']}")
    print(f"Alphabet: {alphabet}")
    print(f"States: {dfa_rep['states_list']}")
    print(f"Initial state: {initial_state_val} (idx {current_state_idx})")
    print(f"Accept states: {config['parsed_fsm']['accept_states']}")
    print(f"Input string: '{''.join(input_string)}'")
    print("--- Simulation ---")
    print(f"Start state: {idx_to_state[current_state_idx]}")

    current_state_vector = np.zeros(n_states, dtype=int)
    current_state_vector[current_state_idx] = 1


    for step, symbol in enumerate(input_string):
        if symbol not in M_sigma_matrices:
            print(f"Symbol '{symbol}' at step {step} not in FSM alphabet. String rejected.")
            return False, traversed_states_actual
        
        M = M_sigma_matrices[symbol]
        
        # M_sigma(i,j)=1 iff delta_w(q_j)=q_i. State vector is column.
        # next_state_vector = M @ current_state_vector
        next_state_vector = np.dot(M, current_state_vector)
        
        if np.sum(next_state_vector) != 1:
            # This implies non-determinism or incomplete DFA transition (e.g., to a trap state not modeled)
            # Or the M matrix column for current_state_vector didn't sum to 1.
            current_physical_state_idx = np.argmax(current_state_vector) # find current state index
            print(f"Error/Warning: Transition for state {idx_to_state[current_physical_state_idx]} on symbol '{symbol}' did not lead to a single next state.")
            print(f"M_sigma for '{symbol}':\n{M}")
            print(f"Current state vector: {current_state_vector}")
            print(f"Resulting next_state_vector: {next_state_vector}")
            
            # Try to find where the '1' is in the M column for the current state
            # This is a more direct way to find the single next state for a DFA
            current_state_physical_idx_direct = -1
            for i in range(len(current_state_vector)):
                if current_state_vector[i] == 1:
                    current_state_physical_idx_direct = i
                    break
            
            if current_state_physical_idx_direct != -1:
                target_col_in_M = M[:, current_state_physical_idx_direct]
                next_state_indices = np.where(target_col_in_M == 1)[0]
                if len(next_state_indices) == 1:
                    current_state_idx = next_state_indices[0]
                    current_state_vector = np.zeros(n_states, dtype=int)
                    current_state_vector[current_state_idx] = 1
                    print(f"Transition: {idx_to_state[current_state_physical_idx_direct]} --{symbol}--> {idx_to_state[current_state_idx]}")
                else:
                    print("String rejected due to ambiguous or missing transition.")
                    return False, traversed_states_actual
            else: # Should not happen if current_state_vector is one-hot
                 print("String rejected due to invalid current state vector.")
                 return False, traversed_states_actual

        else: # Valid one-hot next_state_vector
            current_state_vector = next_state_vector
            current_state_idx = np.argmax(current_state_vector)
            print(f"Symbol: '{symbol}' -> New state: {idx_to_state[current_state_idx]}")

        traversed_states_actual.append(idx_to_state[current_state_idx])

    # Final check: is the last state an accept state?
    # Dot product of final state vector with omega vector
    is_accepted = bool(np.dot(current_state_vector, omega_vec) > 0)
    
    print("--- End of Simulation ---")
    if is_accepted:
        print(f"Input string '{''.join(input_string)}' IS ACCEPTED.")
    else:
        print(f"Input string '{''.join(input_string)}' IS REJECTED.")
    print(f"Path: {' -> '.join(map(str,traversed_states_actual))}")
    
    return is_accepted, traversed_states_actual


if __name__ == '__main__':
    # --- Example Usage of the Simulator ---
    conceptual_model_path = "rwkv7_fsm_conceptual_model.pth" # Ensure this file was created by the constructor

    if not_conceptual_model_path_exists := True: # Simplified check for example
        try:
            # 1. Load the conceptual model configuration
            # Ensure 'rwkv7_fsm_conceptual_model.pth' exists from running the constructor script
            # If it doesn't, this will fail. For a standalone run, you might need to ensure
            # the constructor part of the other script has run and produced this file.
            import os
            if not os.path.exists(conceptual_model_path):
                print(f"Error: Conceptual model file '{conceptual_model_path}' not found.")
                print("Please run the RWKV7FSMConstructor script first to generate it.")
                exit()

            config_data = load_conceptual_model(conceptual_model_path)

            # 2. Define an input string to test
            # Example: for the FSM defined in the unit tests (0->1 on 'a', 0->0 on 'b', etc.)
            # Initial: 0, Accept: 1
            # "a" -> 0 --a--> 1 (Accept)
            # "ab" -> 0 --a--> 1 --b--> 1 (Accept)
            # "b" -> 0 --b--> 0 (Reject)
            # "aa" -> 0 --a--> 1 --a--> 0 (Reject)

            

            while True:
                test_str = input("\nEnter a string to test (or 'q'/'quit' to exit): ").strip()
                if test_str.lower() in ['q', 'quit']:
                    print("Exiting simulator.")
                    break
                if not test_str:
                    print("Input cannot be empty. Try again.")
                    continue
                
                print(f"\nTesting string: '{test_str}'")
                accepted, path = simulate_fsm_from_config(config_data, list(test_str))
                print(f"Accepted: {accepted}, Path: {path}")
                print("-" * 20)

        except Exception as e:
            print(f"An error occurred during simulation: {e}")
            import traceback
            traceback.print_exc()

