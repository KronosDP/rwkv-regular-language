import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraMLP(nn.Module):
    """LoRA MLP as described in the RWKV-7 paper."""
    def __init__(self, d_model, d_lora_hidden, bias=True, activation_fn=None):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_lora_hidden, bias=bias)
        self.fc2 = nn.Linear(d_lora_hidden, d_model, bias=bias)
        self.activation_fn = activation_fn

    def forward(self, x):
        h = self.fc1(x)
        if self.activation_fn:
            h = self.activation_fn(h)
        return self.fc2(h)

class RWKV_TimeMix(nn.Module):
    """RWKV-7 Time Mixing Block."""
    def __init__(self, d_model, head_size, num_heads, layer_id,
                 lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size # N
        self.num_heads = num_heads # H
        self.layer_id = layer_id
        
        # Token shift interpolation factors (mu)
        self.mu_r = nn.Parameter(torch.rand(d_model))
        self.mu_k = nn.Parameter(torch.rand(d_model))
        self.mu_v = nn.Parameter(torch.rand(d_model)) # For v_t,l and v_t,c (if layer_id == 0)
        self.mu_d = nn.Parameter(torch.rand(d_model)) # For decay precursor d_t
        self.mu_a = nn.Parameter(torch.rand(d_model)) # For ICLR precursor a_t
        self.mu_g = nn.Parameter(torch.rand(d_model)) # For RWKV gate g_t

        # Linear projections for R, K, V
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v_current_layer = nn.Linear(d_model, d_model, bias=False) # W_v for v_t,l (current layer precursor)

        # LoRA MLPs for data-dependent parameters
        self.decay_lora = LoraMLP(d_model, lora_dim_w, bias=True, activation_fn=torch.tanh)
        self.iclr_lora = LoraMLP(d_model, lora_dim_a, bias=True) # Outputs pre-sigmoid for a_t
        self.value_residual_gate_lora = LoraMLP(d_model, lora_dim_v, bias=True) # Outputs pre-sigmoid for gate in v_t lerp
        self.gate_lora = LoraMLP(d_model, lora_dim_g, bias=False, activation_fn=torch.sigmoid) # For g_t (RWKV gate)

        # Learnable parameters for key modifications
        self.removal_key_multiplier_xi = nn.Parameter(torch.randn(d_model)) # xi for kappa_t (eq.6)
        self.iclr_mix_alpha = nn.Parameter(torch.rand(d_model)) # alpha for k_bar_t (eq.7)

        # For WKV bonus (eq.20) - rho is per head
        self.bonus_multiplier_rho = nn.Parameter(torch.randn(num_heads, self.head_size))

        self.W_o = nn.Linear(d_model, d_model, bias=False) # Output projection
        self.ln_out_tm = nn.GroupNorm(num_heads, d_model) # GroupNorm after WKV readout (as per Fig 11 / pseudocode)

    def forward(self, x, v_prime_c, shift_state_prev, wkv_state_prev):
        B, T, C = x.shape
        H = self.num_heads
        N = self.head_size

        # Token shift (eq.3)
        if shift_state_prev is None: shift_state_prev = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        x_shifted = torch.cat([shift_state_prev, x[:, :-1]], dim=1)
        current_shift_state = x[:, -1:, :].clone()

        # Interpolated inputs for R, K, V, D, A, G
        x_r_lerp = x + (x_shifted - x) * self.mu_r
        x_k_lerp = x + (x_shifted - x) * self.mu_k
        x_v_lerp = x + (x_shifted - x) * self.mu_v
        x_d_lerp = x + (x_shifted - x) * self.mu_d
        x_a_lerp = x + (x_shifted - x) * self.mu_a
        x_g_lerp = x + (x_shifted - x) * self.mu_g

        # Weight Preparation (eq.4-14)
        r_vec = self.W_r(x_r_lerp)    # (B,T,C)
        k_vec = self.W_k(x_k_lerp)    # (B,T,C)
        v_prime_l = self.W_v_current_layer(x_v_lerp) # v_t,l (eq.9)

        d_lora_out = self.decay_lora(x_d_lerp) # Output of loramlp_d for d_t
        w_vec = torch.exp(-torch.exp(torch.tensor(-0.5, device=x.device, dtype=x.dtype)) * torch.sigmoid(d_lora_out.float())).type_as(x) # w_t (eq.12)

        a_vec = torch.sigmoid(self.iclr_lora(x_a_lerp).float()).type_as(x) # a_t (eq.4)
        g_vec = self.gate_lora(x_g_lerp) # g_t (eq.14), sigmoid is inside LoraMLP

        # Value v_t computation (eq.10)
        # v_prime_c is the value precursor from layer 0 (x_emb_shifted @ W_v_emb_level)
        _v_prime_c_to_use = self.W_v_current_layer(x_v_lerp) if self.layer_id == 0 else v_prime_c
        
        value_residual_gate = torch.sigmoid(self.value_residual_gate_lora(x_v_lerp).float()).type_as(x) # eq.8
        # v_t = lerp(v_t,c, v_t,l, gate) = v_t,c + (v_t,l - v_t,c) * gate
        v_vec = _v_prime_c_to_use + (v_prime_l - _v_prime_c_to_use) * value_residual_gate


        kappa_vec = k_vec * self.removal_key_multiplier_xi # kappa_t = k_t * xi (eq.6)
        # k_bar_t = k_t * lerp(1, a_t, alpha) = k_t * (1 + (a_t - 1) * alpha) (eq.7)
        k_bar_vec = k_vec * (1 + (a_vec - 1) * self.iclr_mix_alpha)

        # Reshape for multi-head operations
        r_head = r_vec.view(B, T, H, N)
        w_head = w_vec.view(B, T, H, N)
        k_bar_head = k_bar_vec.view(B, T, H, N) # Replacement key
        v_head = v_vec.view(B, T, H, N)
        kappa_head = kappa_vec.view(B, T, H, N) # Removal key precursor
        a_head = a_vec.view(B, T, H, N)         # ICLR per head

        kappa_hat_head = F.normalize(kappa_head, p=2, dim=-1) # Normalized removal key (eq.15)

        # WKV state evolution (Recurrence) (eq.16, 17)
        if wkv_state_prev is None:
            wkv_state_prev = torch.zeros(B, H, N, N, device=x.device, dtype=x.dtype)

        wkv_readouts_over_time = []
        current_wkv_state = wkv_state_prev # (B,H,N,N)

        for t_step in range(T):
            rt, wt, kt_bar, vt, kappat_hat, at = \
                r_head[:,t_step], w_head[:,t_step], k_bar_head[:,t_step], v_head[:,t_step], \
                kappa_hat_head[:,t_step], a_head[:,t_step] # Each is (B,H,N)

            # Transition Matrix G_t (eq.19)
            # G_t = diag(w_t) - kappa_hat_t^T (a_t . kappa_hat_t)
            term_inside_paren = at * kappat_hat # Element-wise (B,H,N)
            # outer_prod_term is kappa_hat_t (as col vec) times term_inside_paren (as row vec)
            outer_prod_term = kappat_hat.unsqueeze(-1) * term_inside_paren.unsqueeze(-2) # (B,H,N,N)
            diag_wt = torch.diag_embed(wt) # (B,H,N,N)
            G_t = diag_wt - outer_prod_term

            # Additive term: v_t^T . k_bar_t (outer product)
            vk_outer_prod = vt.unsqueeze(-1) * kt_bar.unsqueeze(-2) # (B,H,N,N)
            
            current_wkv_state = current_wkv_state @ G_t + vk_outer_prod # (B,H,N,N)

            # Readout: r_t @ WKV_state (using einsum for clarity as in Appendix H)
            # r_t (B,H,N), current_wkv_state (B,H,N,N) -> readout (B,H,N)
            wkv_readout_t = torch.einsum('bhn,bhmn->bhm', rt, current_wkv_state)
            wkv_readouts_over_time.append(wkv_readout_t)
        
        final_wkv_state_to_pass = current_wkv_state.clone()

        # Stack readouts, reshape to (B,T,C)
        p_prime = torch.stack(wkv_readouts_over_time, dim=1).view(B, T, C) # (eq.21 part 1, before norm)
        
        # Normalize p_prime (eq.21 part 1)
        p_prime_norm = self.ln_out_tm(p_prime.transpose(1,2).contiguous()).transpose(1,2).contiguous()

        # WKV Bonus u_t (eq.20)
        # u_t = (r_t . (rho . k_bar_t)^T) v_t
        # Interpreting (rho . k_bar_t) as element-wise product, then (r_t . that_vector) as inner product (scalar)
        # Then scalar * v_t. This makes u_t a vector (B,T,H,N).
        rho_expanded = self.bonus_multiplier_rho.unsqueeze(0).unsqueeze(0) # (1,1,H,N)
        term_rho_k_bar = rho_expanded * k_bar_head # (B,T,H,N)
        # Inner product of r_head and term_rho_k_bar for each head
        scalar_per_head = (r_head * term_rho_k_bar).sum(dim=-1, keepdim=True) # (B,T,H,1)
        bonus_u_head = scalar_per_head * v_head # (B,T,H,N)
        bonus_u = bonus_u_head.view(B,T,C) # Reshape to (B,T,C)

        p_t = p_prime_norm + bonus_u # (eq.21 part 2)
        
        # Final gating and projection (eq.22)
        output = self.W_o(g_vec * p_t) # g_vec is (B,T,C)
        
        return output, current_shift_state, final_wkv_state_to_pass


class RWKV_ChannelMix(nn.Module):
    """RWKV-7 Channel Mixing Block (Simplified FFN)."""
    def __init__(self, d_model, hidden_dim_multiplier=4):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * hidden_dim_multiplier
        
        self.mu_k_prime = nn.Parameter(torch.rand(d_model)) # For token shift (eq.23)
        
        self.W_k_prime = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.W_v_prime = nn.Linear(self.hidden_dim, d_model, bias=False)

    def forward(self, x, shift_state_prev):
        B, T, C = x.shape
        if shift_state_prev is None: shift_state_prev = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        x_shifted = torch.cat([shift_state_prev, x[:, :-1]], dim=1)
        current_shift_state = x[:, -1:, :].clone()

        # k_t_prime = lerp(x_t_prime, x_t-1_prime, mu_k_prime) W_k_prime (eq.23)
        x_k_prime_lerp = x + (x_shifted - x) * self.mu_k_prime
        k_prime = self.W_k_prime(x_k_prime_lerp)
        
        # o_t_prime = ReLU(k_t_prime)^2 W_v_prime (eq.24)
        output = self.W_v_prime(torch.relu(k_prime)**2)
        return output, current_shift_state


class RWKV_Block(nn.Module):
    """A single RWKV-7 Block with TimeMix and ChannelMix."""
    def __init__(self, d_model, head_size, num_heads, layer_id, ffn_hidden_multiplier,
                 lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g):
        super().__init__()
        self.layer_id = layer_id # To inform TimeMix
        
        self.ln_tm_in = nn.LayerNorm(d_model)
        self.tm = RWKV_TimeMix(d_model, head_size, num_heads, layer_id,
                               lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g)
        
        self.ln_cm_in = nn.LayerNorm(d_model)
        self.cm = RWKV_ChannelMix(d_model, ffn_hidden_multiplier)

    def forward(self, x, v_prime_c, # Value precursor from layer 0, for TimeMix
                tm_shift_state_prev, tm_wkv_state_prev, cm_shift_state_prev):
        
        # Time Mixing part
        tm_input = self.ln_tm_in(x)
        dx_tm, next_tm_shift_state, next_tm_wkv_state = self.tm(
            tm_input, v_prime_c, tm_shift_state_prev, tm_wkv_state_prev
        )
        x = x + dx_tm # Residual connection

        # Channel Mixing part
        cm_input = self.ln_cm_in(x)
        dx_cm, next_cm_shift_state = self.cm(cm_input, cm_shift_state_prev)
        x = x + dx_cm # Residual connection
        
        return x, next_tm_shift_state, next_tm_wkv_state, next_cm_shift_state


class RWKV7_Model_Classifier(nn.Module):
    """Full RWKV-7 Model adapted for sequence classification."""
    def __init__(self, d_model, n_layer, vocab_size, head_size=32, ffn_hidden_multiplier=4,
                 lora_dim_w=64, lora_dim_a=64, lora_dim_v=32, lora_dim_g=128):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.head_size = head_size
        if d_model % head_size != 0: raise ValueError("d_model must be divisible by head_size")
        self.num_heads = d_model // head_size
        self.vocab_size = vocab_size # Needed for embedding layer

        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # For v_prime_c (value precursor from layer 0, used in all TimeMix blocks)
        self.mu_v_for_v_prime_c = nn.Parameter(torch.rand(d_model)) # Token shift for v_prime_c input
        self.W_v_for_v_prime_c = nn.Linear(d_model, d_model, bias=False) # Projection for v_prime_c

        self.ln_pre_embed = nn.LayerNorm(d_model) # LayerNorm after embedding (Fig 1: LayerNorm -> Embedding -> LayerNorm)
                                                # Paper Fig 1 shows: Embedding -> LN -> Block. Let's follow Fig 1.

        self.blocks = nn.ModuleList([
            RWKV_Block(d_model, head_size, self.num_heads, i, ffn_hidden_multiplier,
                       lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g)
            for i in range(n_layer)
        ])

        self.ln_post_blocks = nn.LayerNorm(d_model) # Final LayerNorm before classification head
        self.classification_head = nn.Linear(d_model, 1) # Outputs a single logit for binary classification

    def forward(self, input_ids, states_list_prev=None): # input_ids: (B, T)
        B, T = input_ids.shape
        
        if T == 0: # Handle empty sequences if they can occur
            # Return a fixed logit (e.g., 0) or based on some bias.
            # This depends on how empty strings should be classified (e.g., "" is (ab)*, so label 1)
            # The classification head needs some input.
            # For now, let's assume T > 0 due to collate_fn padding empty to length 1.
            # If an empty string genuinely needs to be processed as T=0, specific logic is needed.
            # The current collate_fn pads empty strings to length 1 using VOCAB['<pad>'].
            pass


        x_emb = self.embedding(input_ids) # (B, T, C)
        
        # Calculate v_prime_c once from the initial token-shifted embeddings
        # This v_prime_c is then passed to all TimeMix blocks.
        # Token shift for v_prime_c input (based on x_emb)
        initial_shift_state_for_vpc = torch.zeros(B, 1, self.d_model, device=x_emb.device, dtype=x_emb.dtype)
        x_emb_shifted_for_vpc = torch.cat([initial_shift_state_for_vpc, x_emb[:, :-1]], dim=1)
        x_v_lerp_for_vpc = x_emb + (x_emb_shifted_for_vpc - x_emb) * self.mu_v_for_v_prime_c
        v_prime_c = self.W_v_for_v_prime_c(x_v_lerp_for_vpc) # (B, T, C)

        current_x = self.ln_pre_embed(x_emb) # LayerNorm after embedding

        if states_list_prev is None:
            states_list_prev = []
            for _ in range(self.n_layer):
                states_list_prev.append({
                    'tm_shift_state': torch.zeros(B, 1, self.d_model, device=current_x.device, dtype=current_x.dtype),
                    'tm_wkv_state': torch.zeros(B, self.num_heads, self.head_size, self.head_size, device=current_x.device, dtype=current_x.dtype),
                    'cm_shift_state': torch.zeros(B, 1, self.d_model, device=current_x.device, dtype=current_x.dtype)
                })
        
        next_states_list_to_return = []
        
        for i in range(self.n_layer):
            block_state_prev = states_list_prev[i]
            current_x, next_tm_ss, next_tm_ws, next_cm_ss = self.blocks[i](
                current_x, v_prime_c, # Pass the same v_prime_c (from initial embedding) to all layers
                block_state_prev['tm_shift_state'],
                block_state_prev['tm_wkv_state'],
                block_state_prev['cm_shift_state']
            )
            next_states_list_to_return.append({
                'tm_shift_state': next_tm_ss,
                'tm_wkv_state': next_tm_ws,
                'cm_shift_state': next_cm_ss
            })
        
        final_x_representation = self.ln_post_blocks(current_x)
        
        # Use the hidden state of the *last actual token* for classification.
        # If using padding, we need to find the actual length of each sequence in the batch.
        # For simplicity here, assuming collate_fn pads, and we take the state at T-1.
        # A more robust way for padded sequences: gather last non-pad token's state.
        # Current collate_fn pads empty strings to length 1 with <pad>.
        # If input_ids is (B, T), last_token_hidden_state is (B, C)
        last_token_hidden_state = final_x_representation[:, -1, :] 
        
        class_logits = self.classification_head(last_token_hidden_state) # (B, 1)
        
        return class_logits, next_states_list_to_return
