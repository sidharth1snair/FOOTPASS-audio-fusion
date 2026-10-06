import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared between the dataset (which must build a wide-enough audio window
# around each video clip) and the model (which shifts within that window).
# Must stay in sync with the EventConditionedAudioHead's max_abs_tau used
# for TAAD fusion, so a class's learned tau can never point outside the
# window handed to it.
TAAD_AUDIO_MARGIN = 50


class AudioEncoder(nn.Module):
    """Small 1D temporal CNN over (crowd_energy, commentary_energy) -> per-frame embedding."""

    def __init__(self, in_channels=2, hidden=32, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=9, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, embed_dim, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, in_channels, T)
        return self.net(x)  # (B, embed_dim, T)


class EventConditionedAudioHead(nn.Module):
    """
    Learns one temporal offset tau_c (in frames) per action class. For every
    frame t, samples the audio embedding at (t + tau_c) and projects it to a
    per-class logit. Sampling uses grid_sample so tau is differentiable.
    Class 0 (background) is a fixed reference logit of 0.
    """

    def __init__(self, embed_dim, num_action_classes=8, init_tau=0.0, max_abs_tau=100.0):
        super().__init__()
        self.num_action_classes = num_action_classes
        self.max_abs_tau = max_abs_tau
        self.tau = nn.Parameter(torch.full((num_action_classes,), float(init_tau)))
        self.weight = nn.Parameter(torch.randn(num_action_classes, embed_dim) * (embed_dim ** -0.5))
        self.bias = nn.Parameter(torch.zeros(num_action_classes))

    def _shift_all_classes(self, embed, offset=0, output_length=None):
        """
        embed: (B, D, T_in) audio embedding, where index 0 corresponds to
        `offset` frames before the region of interest (e.g. a video clip
        starting at absolute frame sf needs embed indexed from sf - offset,
        so a positive tau can look forward past the clip's own end).

        For each output position t in [0, output_length), samples embed at
        (offset + t + tau_c) for every class c.
        """

        B, D, T_in = embed.shape
        C = self.num_action_classes
        T_out = output_length if output_length is not None else T_in
        device, dtype = embed.device, embed.dtype

        tau = self.tau.clamp(-self.max_abs_tau, self.max_abs_tau)  # (C,)

        t_idx = torch.arange(T_out, device=device, dtype=dtype)               # (T_out,)
        pos = offset + t_idx.unsqueeze(0) + tau.unsqueeze(1)                  # (C,T_out)
        norm_pos = (pos / max(T_in - 1, 1)) * 2 - 1                           # (C,T_out), grid_sample x-coord

        x = embed.unsqueeze(1).expand(B, C, D, T_in).reshape(B * C, D, 1, T_in)

        grid = torch.zeros(B * C, 1, T_out, 2, device=device, dtype=dtype)
        grid_x = norm_pos.unsqueeze(0).expand(B, C, T_out).reshape(B * C, T_out)
        grid[:, 0, :, 0] = grid_x
        grid[:, 0, :, 1] = 0.0

        shifted = F.grid_sample(x, grid, mode='bilinear', align_corners=True, padding_mode='border')  # (B*C,D,1,T_out)

        return shifted.squeeze(2).reshape(B, C, D, T_out)

    def forward(self, embed, offset=0, output_length=None):
        # embed: (B, D, T_in) -> logits: (B, 1+C, T_out)
        B, D, T_in = embed.shape
        T_out = output_length if output_length is not None else T_in

        shifted = self._shift_all_classes(embed, offset=offset, output_length=T_out)  # (B, C, D, T_out)
        class_logits = torch.einsum('bcdt,cd->bct', shifted, self.weight) + self.bias.view(1, -1, 1)  # (B,C,T_out)

        bg_logit = torch.zeros(B, 1, T_out, device=embed.device, dtype=embed.dtype)
        return torch.cat([bg_logit, class_logits], dim=1)  # (B, 1+C, T_out)

    def tau_frames(self):
        return self.tau.detach().clamp(-self.max_abs_tau, self.max_abs_tau).cpu().numpy()

    def tau_seconds(self, fps=25):
        return self.tau_frames() / fps


class AudioOnlyEventDetector(nn.Module):
    """
    Standalone audio-only event detector: encodes (crowd, commentary) energy
    streams and predicts a per-frame, per-class logit using the learned
    per-class temporal lag. Meant as a prototype/ablation for the learned-tau
    mechanism before it's wired into the full TAAD+DST fusion.
    """

    def __init__(self, in_channels=2, hidden=32, embed_dim=32, num_action_classes=8, max_abs_tau=100.0):
        super().__init__()
        self.encoder = AudioEncoder(in_channels, hidden, embed_dim)
        self.head = EventConditionedAudioHead(embed_dim, num_action_classes, max_abs_tau=max_abs_tau)

    def forward(self, x):
        # x: (B, in_channels, T)
        embed = self.encoder(x)
        return self.head(embed)  # (B, 1+num_action_classes, T)
