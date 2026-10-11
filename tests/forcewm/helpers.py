import torch
from torch import nn
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
def model(layers=3):
    torch.manual_seed(7)
    cfg = dict(hidden_dim=16, action_dim=7, ffn_dim=32, text_dim=12, freq_dim=8,
               eps=1e-6, num_heads=2, attn_head_dim=8, num_layers=layers)
    action = ActionDiT(**cfg)
    video = WanVideoDiT(hidden_dim=16, in_dim=48, ffn_dim=32, out_dim=48,
        text_dim=12, freq_dim=8, eps=1e-6, patch_size=(1, 1, 1), num_heads=2,
        attn_head_dim=8, num_layers=layers, has_image_input=False,
        seperated_timestep=True, video_attention_mask_mode="first_frame_causal")
    mot = MoT({"video": video, "action": action})
    vae = nn.Module()
    vae.temporal_downsample_factor = 4
    return FastWAM(video, action, mot, vae,
                   text_dim=12, proprio_dim=8)


def sample(batch=2):
    gen = torch.Generator().manual_seed(101)
    return {"video_latents": torch.randn(batch, 48, 3, 2, 2, generator=gen),
        "video_num_frames": 9, "action": torch.randn(batch, 32, 7, generator=gen),
        "proprio": torch.randn(batch, 32, 8, generator=gen),
        "context": torch.randn(batch, 3, 12, generator=gen),
        "context_mask": torch.tensor([[True, False, True]]).expand(batch, -1).clone(),
        "image_is_pad": torch.zeros(batch, 9, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch, 32, dtype=torch.bool),
        "history": torch.randn(batch, 32, 13, generator=gen),
        "history_is_pad": torch.zeros(batch, 32, dtype=torch.bool),
        "history_channel_valid": torch.ones(batch, 32, 13, dtype=torch.bool),
        "force_is_pad": torch.zeros(batch, 32, dtype=torch.bool),
        "wrench_future": torch.randn(batch, 32, 6, generator=gen),
        "wrench_valid": torch.ones(batch, 32, 6, dtype=torch.bool)}
