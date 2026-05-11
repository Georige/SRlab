"""MorphoSR: Neural Cellular Automata for super-resolution.

Each pixel is a "cell" with 16-channel internal state (RGB + hidden messengers).
A shared, tiny update rule is applied repeatedly (15-30 steps), each step:
  1. Perception: Sobel gradients + learnable depthwise conv (local neighborhood)
  2. Update: 2-layer 1×1 Conv MLP → ΔS (per-pixel, no spatial mixing)
  3. Stochastic: random mask, ~50% cells update per step

The entire model is ~5.4K parameters for the core NCA rule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Pre-encoder: lightweight LR denoiser for real-world input
# ============================================================

class ResBlock(nn.Module):
    """Residual block: Conv → SiLU → Conv, residual connection."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.silu(self.conv1(x)))


class PreEncoder(nn.Module):
    """Lightweight CNN that denoises real LR images at LOW resolution.

    Structure: Conv 3→ch + N×ResBlock(ch) + Conv ch→3, with residual connection.
    Output = input + encoder(input) — learns a denoising residual.

    Args:
        base_ch: internal channel count (default 32)
        n_blocks: number of ResBlocks (default 4)
    """

    def __init__(self, base_ch=32, n_blocks=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(base_ch, 3, 3, padding=1)

        # Zero-init tail for stable start (output ≈ input initially)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        """x: [B, 3, h, w] real LR image in [-1, 1], returns denoised version."""
        return x + self.tail(self.blocks(self.head(x)))


# ============================================================
# MorphoSR: NCA cell update rule
# ============================================================

class MorphoSR(nn.Module):
    """Neural Cellular Automata for super-resolution.

    The model defines a per-cell update rule shared across all pixels and all
    NCA steps.  Calling `forward()` runs the full NCA loop.

    Args:
        state_ch:    cell state channels (default 16)
        hidden_ch:   update MLP hidden channels (default 64)
        cond_ch:     conditioning channels, 0 = no conditioning (default 0 for Stage 1)
        use_pre_encoder: add PreEncoder for real LR denoising (Stage 3+)
        pre_encoder_ch: PreEncoder base channels (default 32)
        pre_encoder_n:  number of ResBlocks in PreEncoder (default 4)
    """

    def __init__(self, state_ch=16, hidden_ch=64, cond_ch=0,
                 use_pre_encoder=False, pre_encoder_ch=32, pre_encoder_n=4,
                 inject_cond=False, zero_init=False):
        super().__init__()
        self.state_ch = state_ch
        self.hidden_ch = hidden_ch
        self.cond_ch = cond_ch
        self.use_pre_encoder_flag = use_pre_encoder
        self.inject_cond = inject_cond
        self.zero_init = zero_init

        # ---- Pre-encoder (optional, Stage 3+) ----
        if use_pre_encoder:
            self.pre_encoder = PreEncoder(base_ch=pre_encoder_ch, n_blocks=pre_encoder_n)
        else:
            self.pre_encoder = None

        # ---- Initialization: 1×1 conv 3→state_ch ----
        self.init_conv = nn.Conv2d(3, state_ch, 1)
        if zero_init:
            nn.init.zeros_(self.init_conv.weight)
            nn.init.zeros_(self.init_conv.bias)
            # In residual mode, zero init_conv means initial SR = bicubic + 0 = bicubic

        # ---- Fixed Sobel kernels for gradient perception ----
        sobel_x_1ch = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        sobel_y_1ch = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).view(1, 1, 3, 3)
        # Expand to groups=state_ch: [state_ch, 1, 3, 3]
        self.register_buffer('sobel_x_kernel', sobel_x_1ch.repeat(state_ch, 1, 1, 1))
        self.register_buffer('sobel_y_kernel', sobel_y_1ch.repeat(state_ch, 1, 1, 1))

        # ---- Learnable depthwise conv for local perception ----
        self.perceive_dw = nn.Conv2d(state_ch, state_ch, 3, padding=1, groups=state_ch)

        # ---- Update MLP: 2-layer 1×1 Conv (per-pixel) ----
        # Input: state + sobel_x + sobel_y + perceived + condition
        mlp_in_ch = state_ch * 4 + cond_ch  # 64 or 67
        self.update_mlp = nn.Sequential(
            nn.Conv2d(mlp_in_ch, hidden_ch, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, state_ch, 1),
        )

        # ---- Initialize last layer to zero for stable start ----
        nn.init.zeros_(self.update_mlp[-1].weight)
        nn.init.zeros_(self.update_mlp[-1].bias)

    def _initialize_state(self, x_lr, target_h, target_w):
        """Map LR image to initial cell state, then nearest-neighbor upscale.

        Args:
            x_lr: [B, 3, h, w] low-res input
            target_h, target_w: HR spatial dimensions

        Returns:
            state: [B, state_ch, target_h, target_w]
        """
        if self.pre_encoder is not None:
            x_lr = self.pre_encoder(x_lr)

        state = self.init_conv(x_lr)
        state = F.interpolate(state, size=(target_h, target_w), mode='nearest')
        return state

    def _perceive(self, state):
        """Perception: fixed Sobel gradients + learnable depthwise conv.

        Args:
            state: [B, state_ch, H, W]

        Returns:
            perceived: [B, state_ch*3, H, W] = sobel_x + sobel_y + dwconv
        """
        B, C, H, W = state.shape

        # Apply sobel per-channel: kernel is [C, 1, 3, 3], groups=C
        grad_x = F.conv2d(state, self.sobel_x_kernel, padding=1, groups=C)
        grad_y = F.conv2d(state, self.sobel_y_kernel, padding=1, groups=C)
        dw = self.perceive_dw(state)

        return torch.cat([grad_x, grad_y, dw], dim=1)

    def _update(self, state, perception, condition=None):
        """Single NCA update step.

        Args:
            state:      [B, state_ch, H, W] current cell state
            perception: [B, state_ch*3, H, W] perceived features
            condition:  [B, cond_ch, H, W] or None, environmental scaffold

        Returns:
            new_state:  [B, state_ch, H, W]
            mask was applied internally with p_update probability
        """
        # Concatenate all inputs for the update MLP
        if condition is not None:
            mlp_input = torch.cat([state, perception, condition], dim=1)
        else:
            mlp_input = torch.cat([state, perception], dim=1)

        dstate = self.update_mlp(mlp_input)

        return dstate

    def forward(self, x_lr, n_steps=15, condition=None, p_update=0.5,
                return_progression=False):
        """Run the NCA for n_steps on the given low-res input.

        Args:
            x_lr:     [B, 3, h, w] low-res input
            n_steps:  number of NCA update iterations
            condition: [B, cond_ch, H, W] conditioning scaffold (bicubic-upsampled LR)
                       If None and cond_ch==0, runs unconditioned.
            p_update: probability a cell updates each step (stochasticity).
                      Set to 1.0 for deterministic inference.
            return_progression: if True, also return a list of intermediate RGB frames

        Returns:
            sr:          [B, 3, H, W] final super-resolved image
            progression: list of [B, 3, H, W] intermediate states (if return_progression)
        """
        B = x_lr.shape[0]

        # Determine target size from condition or infer from scale
        if condition is not None:
            _, _, H, W = condition.shape
        else:
            # Estimate from LR size — assume scale=2 (RealSR native)
            h, w = x_lr.shape[2], x_lr.shape[3]
            H, W = h * 2, w * 2

        # Initialize cell state
        state = self._initialize_state(x_lr, H, W)

        # ---- NCA loop ----
        progression = [] if return_progression else None
        save_every = max(1, n_steps // 6) if return_progression else n_steps + 1

        for step in range(n_steps):
            # ---- Environmental injection: overwrite last 3 state channels with scaffold ----
            if self.inject_cond and condition is not None:
                state[:, -3:] = condition

            # Perception
            perception = self._perceive(state)

            # Compute ΔS — skip external condition if already injected into state
            cond_for_update = None if self.inject_cond else condition
            dstate = self._update(state, perception, cond_for_update)

            # Stochastic update mask
            if p_update < 1.0:
                mask = (torch.rand(B, 1, H, W, device=state.device) < p_update).float()
                state = state + mask * dstate
            else:
                state = state + dstate

            # Collect progression frame (RGB channels only)
            if return_progression and (step + 1) % save_every == 0:
                progression.append(state[:, 0:3].clone())

        # Always include final state in progression
        if return_progression:
            if len(progression) == 0 or (n_steps % save_every != 0):
                progression.append(state[:, 0:3].clone())

        # Output: RGB channels (0-2)
        sr = state[:, 0:3]

        if return_progression:
            return sr, progression, state
        return sr, state

    def load_balance_loss(self):
        """Compatibility stub — MorphoSR has no MoE."""
        return torch.tensor(0.0)
