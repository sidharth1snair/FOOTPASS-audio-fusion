import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.ops import roi_align

from models.audio_fusion import AudioEncoder, EventConditionedAudioHead, TAAD_AUDIO_MARGIN


class X3D_TAAD_Baseline(nn.Module):

    """
    Inputs : sequences of T frames, M sequences of ROIs, M sequences of masks,
        and (optionally) a wider audio-energy window + validity mask:
        [(B,3,T,352,640), (B,M,T,5), (B,M,T), (B,2,T+2*margin), (B,)]

    The audio branch predicts a per-class, per-frame correction using a
    learned per-class temporal offset (see models/audio_fusion.py) and adds
    it to every player's logits for that frame (audio is match-level, not
    player-specific). Samples without audio features available pass
    audio_mask=0, which zeroes the correction and falls back to pure video.
    """

    def __init__(self, use_audio=True, audio_margin=TAAD_AUDIO_MARGIN, audio_max_abs_tau=TAAD_AUDIO_MARGIN):
        super().__init__()

        self.x3d = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
        self.x3d_L4 = nn.Sequential(*[self.x3d.blocks[i] for i in range(2)])
        self.up_L32 = nn.Upsample(scale_factor=(1,2,2))
        self.conv_L16_32 = nn.Conv3d(in_channels=288, out_channels=192, kernel_size=(1,3,3), padding='same', bias=False)
        self.bn_L16_32 = nn.BatchNorm3d(192)
        self.up_L16 = nn.Upsample(scale_factor=(1,2,2))
        self.conv_L8_16 = nn.Conv3d(in_channels=240, out_channels=192, kernel_size=(1,3,3), padding='same', bias=False)
        self.bn_L8_16 = nn.BatchNorm3d(192)
        self.avgpool2D = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv1d(in_channels=192, out_channels=512, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc1 = nn.Linear(512,9)

        self.use_audio = use_audio
        self.audio_margin = audio_margin
        if use_audio:
            self.audio_encoder = AudioEncoder(in_channels=2, hidden=32, embed_dim=32)
            self.audio_head = EventConditionedAudioHead(embed_dim=32, num_action_classes=8, max_abs_tau=audio_max_abs_tau)


    def forward(self, in_x):

        if self.use_audio:
            x, roi, mask, audio_energy, audio_mask = in_x
        else:
            x, roi, mask = in_x

        b,c,l,h,w = x.shape

        w = self.x3d_L4(x) # (B, 48, T, 88, 160)
        z = self.x3d.blocks[2](w) # (B, 48, T, 44, 80)
        y = self.x3d.blocks[3](z) # (B, 96, T, 22, 40)
        x = self.x3d.blocks[4](y) # (B, 192, T, 11, 20)
        x = self.up_L32(x) # (B, 192, T, 22, 40)
        x = torch.concat((x,y), dim=1) # (B, 288, T, 22, 40)
        x = self.conv_L16_32(x) # (B, 192, T, 22, 40)
        x = F.gelu(self.bn_L16_32(x)) # (B, 192, T, 22, 40)
        x = self.up_L16(x) # (B, 192, T, 44, 80)
        x = torch.concat((x,z), dim=1) # (B, 240, T, 44, 80)
        x = self.conv_L8_16(x) # (B, 192, T, 44, 80)
        x = F.gelu(self.bn_L8_16(x)) # (B, 192, T, 44, 80)

        _,_,_,fh,fw = x.shape

        x = x.permute(0,2,1,3,4).reshape(-1,192,fh,fw) # (B*T,192,44,80)

        _,M,_,_ = roi.shape
        roi = roi.permute(0,2,1,3).reshape(-1,5) # (B*T*M,5)
        f_num = roi[:,0]
        batch_indices = torch.arange(b).repeat_interleave(l * M).cuda()
        adjusted_frame_numbers = f_num + batch_indices * l
        roi[:,0] = adjusted_frame_numbers # (B*T*M,5)

        x = roi_align(x, roi, (4,2), 0.125) # (B*T*M, 192, 4, 2)
        x = self.avgpool2D(x).squeeze(-1).squeeze(-1).reshape(b,l,M,192).permute(0,2,3,1).reshape(b*M,192,l) # (B*T*M, 192, 1, 1) to (B*M, 192, T)
        x = x*(mask.reshape(b*M,l).unsqueeze(1)) #  (B*M,192,T)
        x = F.gelu(self.bn1(self.conv1(x))) #  (B*M,512,T)

        x = self.fc1(x.permute(0,2,1)) # (B*M,T,9)

        pred = x.reshape(b,M,l,9).permute(0,3,1,2) # (B*M,T,9) to (B,9,M,T)

        if self.use_audio:
            audio_embed = self.audio_encoder(audio_energy)  # (B, D, l+2*margin)
            audio_logits = self.audio_head(audio_embed, offset=self.audio_margin, output_length=l)  # (B,9,T)
            audio_logits = audio_logits * audio_mask.view(b, 1, 1)  # zero out where no audio available
            pred = pred + audio_logits.unsqueeze(2)  # (B,9,M,T) + (B,9,1,T) broadcast over players

        return pred

