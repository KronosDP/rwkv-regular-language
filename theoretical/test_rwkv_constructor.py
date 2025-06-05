import os
import pickle
import unittest

import numpy as np

# Import the classes and functions from the main module
from rwkv_constructor import (TORCH_AVAILABLE, CopyMatrix, DFARepresentation,
                              FSMParser, IdentityMatrix, MatrixFactorizer,
                              RWKV7FSMConstructor, SwapMatrix, WKVParams)

# Import torch if available
if TORCH_AVAILABLE:
    import torch


class TestFSMToRWKV7(unittest.TestCase):
    def setUp(self):
        self.test_fsm_file = "test_fsm_input.txt"
        with open(self.test_fsm_file, "w") as f:
            f.write("0 1 a x\n")
            f.write("0 0 b y\n")
            f.write("1 0 a z\n")
            f.write("1 1 b w\n")
            f.write("0\n")      
            f.write("1\n")      
        
        self.parser = FSMParser()
        self.fsm_data = self.parser.parse(self.test_fsm_file)

    def tearDown(self):
        if os.path.exists(self.test_fsm_file):
            os.remove(self.test_fsm_file)
        if os.path.exists("test_config.pth"):
            os.remove("test_config.pth")

    def test_fsm_parser(self):
        self.assertEqual(self.fsm_data['states'], [0, 1])
        self.assertEqual(self.fsm_data['alphabet'], ['a', 'b'])
        self.assertEqual(self.fsm_data['initial_state'], 0)
        self.assertEqual(self.fsm_data['accept_states'], [1])
        self.assertEqual(len(self.fsm_data['transitions']), 4)
        self.assertIn({'source': 0, 'input': 'a', 'target': 1, 'output': 'x'}, self.fsm_data['transitions'])

    def test_dfa_representation(self):
        dfa = DFARepresentation(self.fsm_data)
        self.assertEqual(dfa.n_states, 2)
        self.assertEqual(dfa.state_to_idx, {0: 0, 1: 1})
        np.testing.assert_array_equal(dfa.alpha, np.array([1, 0]))
        np.testing.assert_array_equal(dfa.omega, np.array([0, 1]))
        expected_M_a = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(dfa.M_sigma['a'], expected_M_a)
        expected_M_b = np.array([[1, 0], [0, 1]]) 
        np.testing.assert_array_equal(dfa.M_sigma['b'], expected_M_b)

    def test_elementary_matrices_construction_and_effect(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        
        # Identity
        G_I_op = IdentityMatrix(3)
        np.testing.assert_array_equal(X @ G_I_op.matrix, X)

        # Swap: G should swap columns of X when XG
        # SwapMatrix(3,0,1) means G has col 0 = e1, col 1 = e0
        s_op = SwapMatrix(3,0,1) 
        expected_G_S_01 = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
        np.testing.assert_array_equal(s_op.matrix, expected_G_S_01)
        X_swapped_actual = X @ s_op.matrix
        expected_X_swapped = np.array([[2.,1.,3.],[5.,4.,6.],[8.,7.,9.]])
        np.testing.assert_array_equal(X_swapped_actual, expected_X_swapped)
        
        # Copy: G should make XG have col_to_replace as X's col_to_copy_from
        # CopyMatrix(3,0,1) means G has col 0 = e1, other cols are I's cols
        c_op = CopyMatrix(3,0,1) # Replace col 0 of X with col 1 of X
        expected_G_C_01 = np.array([[0.,0.,0.],[1.,1.,0.],[0.,0.,1.]]) # Col 0 is e1, Col 1 is e1, Col 2 is e2
        np.testing.assert_array_equal(c_op.matrix, expected_G_C_01)
        X_copied_actual = X @ c_op.matrix
        expected_X_copied = np.array([[2.,2.,3.],[5.,5.,6.],[8.,8.,9.]])
        np.testing.assert_array_equal(X_copied_actual, expected_X_copied)

    def test_matrix_factorizer_identity(self):
        factorizer = MatrixFactorizer()
        n=2
        M_I = np.eye(n)
        ops_I = factorizer.factorize(M_I, n)
        self.assertEqual(len(ops_I), n)
        # Factorizer might produce non-identity ops that cancel out to I, or actual I's
        X_reconstructed = np.eye(n)
        for op in ops_I:
            X_reconstructed = X_reconstructed @ op.matrix
        np.testing.assert_array_equal(X_reconstructed, M_I, "Factorization of Identity failed")

    def test_matrix_factorizer_swap(self):
        factorizer = MatrixFactorizer()
        n=2
        M_S = np.array([[0,1],[1,0]]) # Swap(0,1)
        ops_S = factorizer.factorize(M_S, n)
        self.assertEqual(len(ops_S), n)
        X_reconstructed = np.eye(n)
        for op in ops_S:
            X_reconstructed = X_reconstructed @ op.matrix
        np.testing.assert_array_equal(X_reconstructed, M_S, "Factorization of Swap matrix failed")

    def test_matrix_factorizer_copy(self):
        # M = [[1,1],[0,0]] (state 0->0, state 1->0). Col 0 is e0. Col 1 is e0.
        # Original X=I. Target M. X -> M.
        # M_Copy needs col 1 of X (which is e1) to become e0. Col 0 of X (e0) is fine.
        # So op should be CopyMatrix(n, col_to_replace=1, col_to_copy_from=0)
        # G for this is I with col 1 = e0. [[1,0],[0,0]].
        factorizer = MatrixFactorizer()
        n=2
        M_C = np.array([[1,1],[0,0]], dtype=int) 
        ops_C = factorizer.factorize(M_C, n)
        self.assertEqual(len(ops_C), n)
        X_reconstructed = np.eye(n, dtype=int)
        for op in ops_C:
            X_reconstructed = X_reconstructed @ op.matrix
        np.testing.assert_array_equal(X_reconstructed, M_C, "Factorization of Copy-like matrix failed")

    def test_wkv_params_generation_matches_matrix(self):
        n=2
        # Identity
        id_op_obj = IdentityMatrix(n)
        id_params = WKVParams.get_wkv_params_for_elementary_matrix(id_op_obj, n)
        G_id_reconstructed = WKVParams.construct_G_from_params(id_params['w_vec'], id_params['kappa_hat_vec'], id_params['a_vec'], id_params['c_rwkv'])
        np.testing.assert_array_almost_equal(G_id_reconstructed, id_op_obj.matrix, decimal=5)

        # Swap(0,1)
        swap_op_obj = SwapMatrix(n,0,1)
        swap_params = WKVParams.get_wkv_params_for_elementary_matrix(swap_op_obj, n)
        G_swap_reconstructed = WKVParams.construct_G_from_params(swap_params['w_vec'], swap_params['kappa_hat_vec'], swap_params['a_vec'], swap_params['c_rwkv'])
        np.testing.assert_array_almost_equal(G_swap_reconstructed, swap_op_obj.matrix, decimal=5)

        # Copy (col 0 gets state from col 1) using Lemma 4 interpretation
        # Lemma 4: G for copy X_new[:,x] = X_old[:,y] is from kappa=(ex-ey)/sqrt(2), a_vec=ex
        # This G = I - ex ex^T + ey ex^T.  For x=0, y=1: G e0 = e1.
        # This matrix G maps state 0 to state 1.
        # Expected G for CopyMatrix(n,0,1) (col 0 of X becomes copy of col 1 of X)
        # G itself has its 0th column as e1. So G = [[0,0],[1,1]] (cols are (0,1), (0,1)) if n=2
        # No, G has its 0th col as e_1 (0,1), and 1st col as e_1 (0,1).
        # G = [[0,0],[1,1]] for n=2 where e0=(1,0), e1=(0,1).
        # The CopyMatrix(n,0,1).matrix is [[0,0],[1,1]].
        # WKVParams for this G should be from x=0, y=1
        copy_op_obj = CopyMatrix(n,0,1) # col 0 gets content of col 1
        copy_params = WKVParams.get_wkv_params_for_elementary_matrix(copy_op_obj, n)
        G_copy_reconstructed = WKVParams.construct_G_from_params(copy_params['w_vec'], copy_params['kappa_hat_vec'], copy_params['a_vec'], copy_params['c_rwkv'])
        # Expected G from Lemma 4 for copy x=0 from y=1 is: [[0,0],[1,1]] (maps state 0 to 1, state 1 to 1)
        # The .matrix from CopyMatrix(n,0,1) is I with col_0 = I_col_1, so [[0,0],[1,1]]
        np.testing.assert_array_almost_equal(G_copy_reconstructed, copy_op_obj.matrix, decimal=5)

    def test_save_to_pth(self):
        constructor = RWKV7FSMConstructor(self.test_fsm_file)
        constructor.build()
        output_file = "test_config.pth"
        constructor.save_to_pth(output_file)
        
        self.assertTrue(os.path.exists(output_file))
        
        # Try loading
        loaded_config = None
        if TORCH_AVAILABLE:
            try:
                loaded_config = torch.load(output_file)
            except: # Fallback to pickle if torch.load fails (e.g. if saved with pickle)
                with open(output_file, 'rb') as f:
                    loaded_config = pickle.load(f)
        else:
            with open(output_file, 'rb') as f:
                loaded_config = pickle.load(f)
        
        self.assertIsNotNone(loaded_config)
        self.assertEqual(loaded_config['fsm_source_file'], self.test_fsm_file)
        self.assertEqual(loaded_config['parsed_fsm']['initial_state'], 0)


if __name__ == '__main__':
    print("Running Unit Tests for RWKV Constructor...")
    unittest.main(verbosity=2)
