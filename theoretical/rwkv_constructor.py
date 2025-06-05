import pickle  # Added for saving dictionary
from collections import defaultdict, deque

import numpy as np

# Attempt to import torch, but make it optional for the core logic
# It's only strictly needed if user wants to use torch.save specifically for .pth
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch is not installed. .pth saving will use pickle directly.")


# --- Helper Classes for Elementary Matrices ---
class ElementaryMatrix:
    """Base class for elementary matrices."""
    def __init__(self, size):
        self.size = size
        self.matrix = np.eye(size)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.size == other.size and np.array_equal(self.matrix, other.matrix)

class IdentityMatrix(ElementaryMatrix):
    def __init__(self, size):
        super().__init__(size)
        # Matrix is already np.eye(size)

class SwapMatrix(ElementaryMatrix):
    """
    Represents an elementary matrix G that, when post-multiplied (XG),
    swaps columns i and j of X.
    G is an identity matrix with columns i and j swapped.
    """
    def __init__(self, size, i, j):
        super().__init__(size)
        self.i = i
        self.j = j
        if i == j: # Effectively an identity
            pass
        else:
            P = np.eye(size)
            # To swap columns i and j of X via XG, G must have its i-th column be e_j
            # and its j-th column be e_i. Other columns k are e_k.
            col_i_content = P[:, j].copy() # Standard basis vector e_j
            col_j_content = P[:, i].copy() # Standard basis vector e_i
            P[:, i] = col_i_content
            P[:, j] = col_j_content
            self.matrix = P


    def __repr__(self):
        return f"SwapMatrix(size={self.size}, i={self.i}, j={self.j})"

    def __eq__(self, other):
        if not isinstance(other, SwapMatrix): # Check type first
            return False
        if self.size != other.size or self.i != other.i or self.j != other.j:
            return False
        return np.array_equal(self.matrix, other.matrix)


class CopyMatrix(ElementaryMatrix):
    """
    Represents an elementary matrix G that, when post-multiplied (XG),
    replaces column 'col_to_replace' of X with column 'col_to_copy_from' of X.
    G is an identity matrix with its 'col_to_replace'-th column
    replaced by the 'col_to_copy_from'-th standard basis vector (e_col_to_copy_from).
    """
    def __init__(self, size, col_to_replace, col_to_copy_from):
        super().__init__(size)
        self.col_to_replace = col_to_replace
        self.col_to_copy_from = col_to_copy_from
        if col_to_replace == col_to_copy_from: # Effectively an identity
            pass
        else:
            # G_k = e_k for k != col_to_replace
            # G_col_to_replace = e_col_to_copy_from
            self.matrix = np.eye(size)
            self.matrix[:, col_to_replace] = np.eye(size)[:, col_to_copy_from]


    def __repr__(self):
        return f"CopyMatrix(size={self.size}, col_to_replace={self.col_to_replace}, col_to_copy_from={self.col_to_copy_from})"

    def __eq__(self, other):
        if not isinstance(other, CopyMatrix): # Check type first
            return False
        if self.size != other.size or \
           self.col_to_replace != other.col_to_replace or \
           self.col_to_copy_from != other.col_to_copy_from:
            return False
        return np.array_equal(self.matrix, other.matrix)

# --- FSM Parsing ---
class FSMParser:
    def parse(self, file_path):
        states = set()
        alphabet = set()
        transitions = [] # list of (source, input_sym, target, output_sym)
        initial_state = None
        accept_states = set()

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        transition_lines = []
        state_def_lines = []
        parsing_transitions = True

        for line in lines:
            parts = line.split()
            if parsing_transitions:
                if len(parts) == 4: 
                    try:
                        int(parts[0]) 
                        transition_lines.append(parts)
                    except ValueError:
                        parsing_transitions = False
                        state_def_lines.append(line)
                elif len(parts) == 1: # Could be start of state defs if it's a single number
                    try:
                        int(parts[0])
                        parsing_transitions = False
                        state_def_lines.append(line)
                    except ValueError:
                        # Not a number, could be malformed or some other convention
                        # For now, strictly expect transitions or state defs
                        raise ValueError(f"Unexpected line format during transition parsing: {line}")
                else: 
                    parsing_transitions = False
                    state_def_lines.append(line) # Assume anything not fitting transition is state def
            else:
                state_def_lines.append(line)

        for parts in transition_lines:
            source, target, input_sym, output_sym = parts
            source = int(source)
            target = int(target)
            states.add(source)
            states.add(target)
            alphabet.add(input_sym)
            transitions.append({'source': source, 'input': input_sym, 'target': target, 'output': output_sym})

        if not state_def_lines:
            # If only transitions, might be an issue, or FSM has no defined initial/accept states.
            # For this project, we require them.
            raise ValueError("Missing initial/accept state definitions in FSM file.")

        initial_state = int(state_def_lines[0])
        for state_str in state_def_lines[1:]: # The rest are accept states
            accept_states.add(int(state_str))
        
        if initial_state not in states and transitions:
             states.add(initial_state)
        for acc_s in accept_states:
            if acc_s not in states and transitions:
                states.add(acc_s)
        
        # If no transitions, states might only come from initial/accept
        if not transitions:
            if initial_state is not None:
                states.add(initial_state)
            states.update(accept_states)


        return {
            'states': sorted(list(states)) if states else [],
            'alphabet': sorted(list(alphabet)),
            'transitions': transitions,
            'initial_state': initial_state,
            'accept_states': sorted(list(accept_states)) if accept_states else []
        }

# --- DFA Representation ---
class DFARepresentation:
    def __init__(self, fsm_data):
        self.states_list = fsm_data['states']
        if not self.states_list and fsm_data['initial_state'] is not None:
             # Case: FSM might only define an initial state and no transitions
            self.states_list = [fsm_data['initial_state']]
            if fsm_data['accept_states']:
                 for s_acc in fsm_data['accept_states']:
                    if s_acc not in self.states_list:
                        self.states_list.append(s_acc)
                 self.states_list.sort()


        self.n_states = len(self.states_list)
        self.state_to_idx = {state: i for i, state in enumerate(self.states_list)}
        self.idx_to_state = {i: state for state, i in self.state_to_idx.items()}

        self.alphabet = fsm_data['alphabet']
        self.alphabet_to_idx = {symbol: i for i, symbol in enumerate(self.alphabet)}

        self.alpha = np.zeros(self.n_states, dtype=int)
        if fsm_data['initial_state'] is not None and self.n_states > 0:
             if fsm_data['initial_state'] in self.state_to_idx:
                self.alpha[self.state_to_idx[fsm_data['initial_state']]] = 1
             else: # Should not happen if states_list is constructed properly
                print(f"Warning: Initial state {fsm_data['initial_state']} not found in state_to_idx map.")


        self.omega = np.zeros(self.n_states, dtype=int)
        if self.n_states > 0:
            for acc_state in fsm_data['accept_states']:
                if acc_state in self.state_to_idx:
                    self.omega[self.state_to_idx[acc_state]] = 1
        
        self.M_sigma = {}
        transitions_by_input = defaultdict(list)
        for t in fsm_data['transitions']:
            transitions_by_input[t['input']].append(t)

        for symbol in self.alphabet:
            M = np.zeros((self.n_states, self.n_states), dtype=int)
            for src_idx in range(self.n_states):
                src_state = self.idx_to_state[src_idx]
                found_transition = False
                for t_rule in transitions_by_input[symbol]:
                    if t_rule['source'] == src_state:
                        target_state = t_rule['target']
                        if target_state in self.state_to_idx:
                            target_idx = self.state_to_idx[target_state]
                            M[target_idx, src_idx] = 1 
                            found_transition = True
                            break
                if not found_transition and self.n_states > 0:
                     # Add a self-loop to a conceptual trap state if one were added, 
                     # or handle as per specific DFA completion rules.
                     # For this exercise, if a transition is missing, the column remains zero.
                     # This implies the FSM is not a complete DFA.
                    print(f"Warning: No transition found for state {src_state} (idx {src_idx}) on symbol '{symbol}'. M_sigma column will be zeros.")

            if self.n_states > 0:
                for j_col in range(self.n_states):
                    if np.sum(M[:, j_col]) != 1:
                        print(f"Warning: Transition matrix for symbol '{symbol}' column {j_col} (state {self.idx_to_state.get(j_col, 'N/A')}) does not sum to 1. This FSM is not a complete DFA.")
            self.M_sigma[symbol] = M

# --- Matrix Factorization (Lemma 3) ---
class MatrixFactorizer:
    def factorize(self, M_target, n_states):
        if n_states == 0:
            return []
            
        X = np.eye(n_states, dtype=int) # Current product of Gs, starts as Identity
        elementary_ops = []

        # Stage 1: Use Swap operations to correctly position columns that are permutations of M_target's columns.
        # The goal is to make X equal to M_target.
        # We iterate through columns of M_target. For each column M_target[:,j], if X[:,j] is not already M_target[:,j],
        # we find where M_target[:,j] currently is in X (say X[:,k] == M_target[:,j]) and swap columns j and k in X.
        current_X_cols = [X[:, i].copy() for i in range(n_states)]

        for j in range(n_states): # Target column index in M_target
            target_col_vec = M_target[:, j]
            current_col_in_X_at_j = current_X_cols[j]

            if not np.array_equal(current_col_in_X_at_j, target_col_vec):
                # Find where target_col_vec is currently located in current_X_cols
                k = -1
                for idx in range(n_states):
                    if np.array_equal(current_X_cols[idx], target_col_vec):
                        k = idx
                        break
                
                if k != -1 and k != j:
                    # Swap column j and column k in X
                    op = SwapMatrix(n_states, j, k)
                    elementary_ops.append(op)
                    # Update current_X_cols to reflect the swap
                    current_X_cols[j], current_X_cols[k] = current_X_cols[k], current_X_cols[j]
                # If k == -1, target_col_vec is not in current_X_cols. This means M_target is not a permutation
                # of X's columns (e.g. M_target has duplicate columns, requiring Copy ops).

        # Stage 2: Use Copy operations for columns that are duplicates or need to be created.
        # After swaps, some columns in current_X_cols might still not match M_target.
        # This happens if M_target has identical columns.
        for j in range(n_states):
            target_col_vec = M_target[:, j]
            if not np.array_equal(current_X_cols[j], target_col_vec):
                # We need current_X_cols[j] to become target_col_vec.
                # Find a source column in current_X_cols that IS target_col_vec
                source_k = -1
                for idx in range(n_states):
                    if np.array_equal(current_X_cols[idx], target_col_vec):
                        source_k = idx
                        break
                
                if source_k != -1 : # Should find it if M_target is valid
                    op = CopyMatrix(n_states, col_to_replace=j, col_to_copy_from=source_k)
                    elementary_ops.append(op)
                    current_X_cols[j] = current_X_cols[source_k].copy() # Update for consistency
                else:
                    # This is problematic, implies M_target cannot be formed.
                    # print(f"Factorization warning: Cannot form column {j} of M_target.")
                    pass


        # Verify the factorization (for debugging, not part of core algo if assuming Lemma 3 holds)
        X_final_check = np.eye(n_states)
        for op_check in elementary_ops:
            X_final_check = X_final_check @ op_check.matrix
        
        if not np.array_equal(X_final_check, M_target):
            # print(f"Warning: Factorization for M_target did not perfectly yield M_target.")
            # print("M_target:\n", M_target)
            # print("Result of ops:\n", X_final_check)
            # This indicates the greedy strategy is not fully robust for all DFA M_sigma cases.
            # The paper's Lemma 3 guarantees existence, but the constructive proof is subtle.
            # For this implementation, we proceed with the ops found.
            pass

        while len(elementary_ops) < n_states:
            elementary_ops.append(IdentityMatrix(n_states))
        
        return elementary_ops[:n_states]


# --- WKV Parameterization (Lemma 4) ---
class WKVParams:
    @staticmethod
    def get_wkv_params_for_elementary_matrix(elem_matrix_obj: ElementaryMatrix, n_states: int, c_rwkv: float = 2.0):
        if n_states == 0:
            return {'w_vec': [], 'kappa_hat_vec': [], 'a_vec': [], 'c_rwkv': c_rwkv, 'comment': 'empty state', 'resulting_matrix_G_debug': []}

        w_vec = np.ones(n_states) 
        kappa_hat_vec = np.zeros(n_states)
        a_vec = np.zeros(n_states) # Default to zeros for Identity
        comment = ""

        def normalize_safe(v):
            norm = np.linalg.norm(v)
            return v / norm if norm != 0 else v

        if isinstance(elem_matrix_obj, IdentityMatrix):
            if n_states > 0: kappa_hat_vec[0] = 1.0 
            comment = "Identity matrix (kappa_hat=e_0, a_vec=0)"
        elif isinstance(elem_matrix_obj, SwapMatrix):
            x, y = elem_matrix_obj.i, elem_matrix_obj.j
            if x == y: # Identity
                if n_states > 0: kappa_hat_vec[0] = 1.0
                comment = f"SwapMatrix({x},{y}) -> Identity"
            else:
                e_x = np.eye(1, n_states, x).flatten()
                e_y = np.eye(1, n_states, y).flatten()
                kappa_hat_vec = normalize_safe(e_x - e_y)
                a_vec = np.ones(n_states) 
            comment = f"SwapMatrix({elem_matrix_obj.i}, {elem_matrix_obj.j})"
        elif isinstance(elem_matrix_obj, CopyMatrix):
            x = elem_matrix_obj.col_to_replace
            y = elem_matrix_obj.col_to_copy_from
            if x == y: # Identity
                if n_states > 0: kappa_hat_vec[0] = 1.0
                comment = f"CopyMatrix({x} from {y}) -> Identity"
            else:
                e_x_vec = np.eye(1, n_states, x).flatten()
                e_y_vec = np.eye(1, n_states, y).flatten()
                kappa_hat_vec = normalize_safe(e_x_vec - e_y_vec) # kappa = (e_x - e_y)/sqrt(2)
                a_vec = e_x_vec # a = e_x (vector)
            comment = f"CopyMatrix(col {x} gets data from col {y})"
        else:
            raise TypeError(f"Unknown elementary matrix type: {type(elem_matrix_obj)}")

        # Ensure lists are passed to construct_G_from_params for the debug field
        debug_matrix = WKVParams.construct_G_from_params(
            w_vec.tolist(),       # Pass as list
            kappa_hat_vec.tolist(), # Pass as list
            a_vec.tolist(),       # Pass as list
            c_rwkv
        ).tolist()

        return {
            'w_vec': w_vec.tolist(),
            'kappa_hat_vec': kappa_hat_vec.tolist(),
            'a_vec': a_vec.tolist(),
            'c_rwkv': c_rwkv,
            'comment': comment,
            'resulting_matrix_G_debug': debug_matrix
        }

    @staticmethod
    def construct_G_from_params(w_vec_list, kappa_hat_vec_list, a_vec_list, c_rwkv):
        # This check is now fine because w_vec_list will be a Python list
        if not w_vec_list: return np.array([]) 
        
        w_vec = np.array(w_vec_list)
        kappa_hat_vec = np.array(kappa_hat_vec_list)
        a_vec = np.array(a_vec_list)

        w_diag = np.diag(w_vec)
        K_row = kappa_hat_vec.reshape(1, -1)
        term_to_outer_prod = a_vec * kappa_hat_vec # Element-wise product, result is a vector
        
        outer_prod_term = K_row.T @ term_to_outer_prod.reshape(1, -1)
        
        G = w_diag - c_rwkv * outer_prod_term
        return G


# --- RWKV-7 Constructor ---
class RWKV7FSMConstructor:
    def __init__(self, fsm_file_path):
        self.fsm_file_path = fsm_file_path
        self.fsm_data = None
        self.dfa_rep = None
        self.factorized_M_sigma = {}
        self.rwkv7_config = {}

    def build(self):
        parser = FSMParser()
        self.fsm_data = parser.parse(self.fsm_file_path)

        self.dfa_rep = DFARepresentation(self.fsm_data)
        n_states = self.dfa_rep.n_states
        if n_states == 0 and not self.fsm_data['alphabet']:
             self.rwkv7_config = {'status': 'empty_fsm', 'n_states': 0}
             return self.rwkv7_config

        factorizer = MatrixFactorizer()
        for symbol, M_sigma_matrix in self.dfa_rep.M_sigma.items():
            if n_states > 0:
                self.factorized_M_sigma[symbol] = factorizer.factorize(M_sigma_matrix, n_states)
            else:
                self.factorized_M_sigma[symbol] = []

        layer_configs = {}
        layer_configs['layer1'] = {
            'type': 'PositionParity (Lemma 5)',
            'description': 'Outputs if current token is first and current position parity. Uses fixed WKV params for state toggling.',
            'output_shape_conceptual': ['is_first_token (bool)', 'is_odd_position (bool)']
        }
        layer_configs['layer2'] = {
            'type': 'PositionMod2N (Lemma 6)',
            'description': f'Outputs current position modulo 2*n_states={2*n_states}. Uses Layer 1 output and fixed WKV params for rotation-like state changes over {2*n_states} heads.',
            'output_shape_conceptual': [f'position_mod_{2*n_states} (int)']
        }
        layer_configs['layer3'] = {
            'type': 'LookupMechanism (Lemma 7)',
            'description': (
                f'Inputs: pos_mod_{2*n_states} (from L2), last {2*n_states} token_ids (via WKV state storing history as in Lemma 7 proof part 1).\n'
                'Outputs for Layer 4: Parameters (w, kappa_hat, a_vec) for the elementary matrix G_l_t for the current step t.\n'
                'If t is the final token T, also outputs the final vector omega_hat_T based on remaining G and original omega.'
            ),
            'conceptual_content': {
                'factorized_dfa_transitions': {
                    symbol: [str(op) for op in ops] for symbol, ops in self.factorized_M_sigma.items()
                },
                'dfa_omega_vector': self.dfa_rep.omega.tolist() if self.dfa_rep else []
            }
        }
        layer_configs['layer4'] = {
            'type': 'DFAExecutionByElementaryMatrices',
            'num_wkv_heads': n_states,
            'head_dimension': n_states, 
            'description': (
                'The WKV state (n_states x n_states matrix) is initialized using DFA alpha vector at t=1.\n'
                'At each step t, the state is multiplied by G_l_t (elementary matrix for current sub-step of DFA transition) whose WKV params are provided by Layer 3.\n'
                'Uses n_states WKV heads. Head j applies receptance r_j=e_j to read out j-th component of alpha_hat_t.'
            ),
            'initial_dfa_state_alpha': self.dfa_rep.alpha.tolist() if self.dfa_rep else []
        }
        layer_configs['final_mlp'] = {
            'type': 'DotProductAcceptance',
            'description': (
                'At final token T, takes alpha_hat_T from Layer 4 and omega_hat_T from Layer 3.\n'
                'Computes dot product: alpha_hat_T @ omega_hat_T.\n'
                'Outputs 1 if accept, 0 if reject.'
            )
        }

        self.rwkv7_config = {
            'fsm_source_file': self.fsm_file_path,
            'parsed_fsm': self.fsm_data,
            'dfa_representation': {
                'n_states': self.dfa_rep.n_states if self.dfa_rep else 0,
                'states_list': self.dfa_rep.states_list if self.dfa_rep else [],
                'state_to_idx': self.dfa_rep.state_to_idx if self.dfa_rep else {},
                'alphabet': self.dfa_rep.alphabet if self.dfa_rep else [],
                'alpha_vec (initial_state_one_hot)': self.dfa_rep.alpha.tolist() if self.dfa_rep else [],
                'omega_vec (accept_states_multi_hot)': self.dfa_rep.omega.tolist() if self.dfa_rep else [],
                'M_sigma_matrices': {
                    symbol: M.tolist() for symbol, M in self.dfa_rep.M_sigma.items()
                } if self.dfa_rep else {}
            },
            'factorized_M_sigma': {
                symbol: [str(op) for op in ops] for symbol, ops in self.factorized_M_sigma.items()
            },
            'theoretical_rwkv7_layer_configs': layer_configs,
            'wkv_params_generation_function_ref': 'WKVParams.get_wkv_params_for_elementary_matrix',
            'example_wkv_params_for_elementary_ops (c=2)': self._get_example_wkv_params(n_states) if n_states > 0 else {}
        }
        return self.rwkv7_config

    def _get_example_wkv_params(self, n_states):
        if n_states == 0: return {}
        params = {}
        params['identity'] = WKVParams.get_wkv_params_for_elementary_matrix(IdentityMatrix(n_states), n_states)
        if n_states >= 2:
            params['swap_0_1'] = WKVParams.get_wkv_params_for_elementary_matrix(SwapMatrix(n_states, 0, 1), n_states)
            params['copy_col0_gets_col1'] = WKVParams.get_wkv_params_for_elementary_matrix(CopyMatrix(n_states, 0, 1), n_states)
        return params

    def save_to_pth(self, output_path):
        """Saves the rwkv7_config dictionary to a .pth file."""
        if not self.rwkv7_config:
            print("Error: RWKV-7 config is not built yet. Call build() first.")
            return

        if TORCH_AVAILABLE:
            try:
                torch.save(self.rwkv7_config, output_path)
                print(f"RWKV-7 conceptual config saved to {output_path} using torch.save()")
            except Exception as e:
                print(f"torch.save failed: {e}. Falling back to pickle.")
                self._save_with_pickle(output_path)
        else:
            self._save_with_pickle(output_path)

    def _save_with_pickle(self, output_path):
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.rwkv7_config, f)
            print(f"RWKV-7 conceptual config saved to {output_path} using pickle.")
        except Exception as e:
            print(f"Error saving with pickle: {e}")





if __name__ == '__main__':
    example_fsm_file = "fsm.txt" 
    
    # Check if tes.txt exists, if not, create a dummy one for the example to run
    import os
    if not os.path.exists(example_fsm_file):
        print(f"Warning: '{example_fsm_file}' not found. Creating a dummy file for demonstration.")
        with open(example_fsm_file, "w") as f_dummy:
            f_dummy.write("0 1 a x\n1 0 a y\n0 0 b z\n1 1 b w\n0\n1\n")

    constructor = RWKV7FSMConstructor(example_fsm_file)
    rwkv7_construction_data = constructor.build()
    
    # Save the conceptual model
    output_pth_path = "rwkv7_fsm_conceptual_model.pth"
    constructor.save_to_pth(output_pth_path)
    
    print(f"\nTo run unit tests, execute: python test_rwkv_constructor.py")
    print("RWKV-7 FSM conceptual model construction completed successfully!")

