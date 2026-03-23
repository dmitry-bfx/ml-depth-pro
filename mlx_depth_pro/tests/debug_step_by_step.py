#!/usr/bin/env python3
"""Step-by-step comparison of PyTorch vs MLX post-ViT pipeline to find divergence."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT


def to_mlx(t: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array, NCHW -> NHWC for 4D."""
    arr = t.detach().cpu().float().numpy()
    a = mx.array(arr)
    if a.ndim == 4:
        a = mx.transpose(a, [0, 2, 3, 1])  # NCHW -> NHWC
    return a


def to_mlx_weight_conv(t: torch.Tensor) -> mx.array:
    """Conv2d weight: PyTorch [O,I,kH,kW] -> MLX [O,kH,kW,I]."""
    return mx.array(t.detach().cpu().float().permute(0, 2, 3, 1).contiguous().numpy())


def to_mlx_weight_convt(t: torch.Tensor) -> mx.array:
    """ConvTranspose2d weight: PyTorch [I,O,kH,kW] -> MLX [O,kH,kW,I]."""
    return mx.array(t.detach().cpu().float().permute(1, 2, 3, 0).contiguous().numpy())


def compare(name: str, pt: torch.Tensor, mlx_arr: mx.array, nchw_to_nhwc=True):
    """Compare PyTorch (NCHW) and MLX (NHWC) tensors."""
    pt_np = pt.detach().cpu().float().numpy()
    mx.eval(mlx_arr)
    mlx_np = np.array(mlx_arr)
    if nchw_to_nhwc and pt_np.ndim == 4:
        pt_np = np.transpose(pt_np, [0, 2, 3, 1])  # NCHW -> NHWC
    diff = np.abs(pt_np - mlx_np)
    print(f"{name}:")
    print(f"  shapes: PT={list(pt.shape)} MLX={list(mlx_arr.shape)}")
    print(f"  PT range:  [{pt_np.min():.6f}, {pt_np.max():.6f}]")
    print(f"  MLX range: [{mlx_np.min():.6f}, {mlx_np.max():.6f}]")
    print(f"  max diff:  {diff.max():.6e}, mean diff: {diff.mean():.6e}")
    if pt_np.max() > 1e-8:
        rel = diff / (np.abs(pt_np) + 1e-8)
        print(f"  max rel:   {rel.max():.6e}")
    return diff.max()


def mlx_conv2d(x, weight, bias=None, stride=(1,1), padding=(0,0)):
    y = mx.conv2d(x, weight, stride=stride, padding=padding)
    if bias is not None:
        y = y + bias
    return y


def mlx_conv_transpose2d(x, weight, bias=None, stride=(2,2)):
    y = mx.conv_transpose2d(x, weight, stride=stride)
    if bias is not None:
        y = y + bias
    return y


def mlx_merge(x, batch_size, steps, padding):
    """Replicate PyTorch merge in MLX (NHWC layout)."""
    idx = 0
    rows = []
    for j in range(steps):
        row_patches = []
        for i in range(steps):
            patch = x[batch_size * idx : batch_size * (idx + 1)]
            h_start, h_end = 0, patch.shape[1]
            w_start, w_end = 0, patch.shape[2]
            if j != 0: h_start = padding
            if j != steps - 1: h_end -= padding
            if i != 0: w_start = padding
            if i != steps - 1: w_end -= padding
            patch = patch[:, h_start:h_end, w_start:w_end, :]
            row_patches.append(patch)
            idx += 1
        rows.append(mx.concatenate(row_patches, axis=2))
    return mx.concatenate(rows, axis=1)


def main():
    print("Loading PyTorch model...")
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = "./checkpoints/depth_pro.pt"
    model, _ = create_model_and_transforms(config, device=torch.device("cpu"))
    model.eval()
    encoder = model.encoder

    torch.manual_seed(42)
    x = torch.randn(1, 3, 1536, 1536)

    print("\n=== Step 1: ViT outputs ===")
    with torch.no_grad():
        x0, x1, x2 = encoder._create_pyramid(x)
        x0_patches = encoder.split(x0, overlap_ratio=0.25)
        x1_patches = encoder.split(x1, overlap_ratio=0.5)
        all_patches = torch.cat([x0_patches, x1_patches, x2], dim=0)

        pt_patch_enc = encoder.patch_encoder(all_patches)
        pt_hook0 = encoder.backbone_highres_hook0.clone()
        pt_hook1 = encoder.backbone_highres_hook1.clone()
        pt_image_enc = encoder.image_encoder(x2)

    print(f"patch_enc: {list(pt_patch_enc.shape)}")
    print(f"hook0: {list(pt_hook0.shape)}")
    print(f"image_enc: {list(pt_image_enc.shape)}")

    print("\n=== Step 2: reshape_feature ===")
    # PyTorch version
    pt_pyramid = encoder.reshape_feature(pt_patch_enc, encoder.out_size, encoder.out_size)
    pt_h0_reshaped = encoder.reshape_feature(pt_hook0, encoder.out_size, encoder.out_size)

    # MLX version
    mlx_patch_enc = mx.array(pt_patch_enc.numpy())
    mlx_tokens = mlx_patch_enc[:, 1:, :]  # remove CLS
    mlx_pyramid = mx.reshape(mlx_tokens, [35, 24, 24, 1024])

    mlx_h0 = mx.array(pt_hook0.numpy())
    mlx_h0_tokens = mlx_h0[:, 1:, :]
    mlx_h0_reshaped = mx.reshape(mlx_h0_tokens, [35, 24, 24, 1024])

    compare("reshape_feature(pyramid)", pt_pyramid, mlx_pyramid)
    compare("reshape_feature(hook0)", pt_h0_reshaped, mlx_h0_reshaped)

    print("\n=== Step 3: merge ===")
    pt_x0_enc = pt_pyramid[:25]
    pt_x0_merged = encoder.merge(pt_x0_enc, batch_size=1, padding=3)

    mlx_x0_enc = mlx_pyramid[:25]
    mlx_x0_merged = mlx_merge(mlx_x0_enc, batch_size=1, steps=5, padding=3)

    compare("merge(x0, 5x5, pad=3)", pt_x0_merged, mlx_x0_merged)

    print("\n=== Step 4: upsample_latent0 (Conv + 3x ConvT) ===")
    pt_h0_25 = pt_h0_reshaped[:25]
    pt_latent0_merged = encoder.merge(pt_h0_25, batch_size=1, padding=3)
    pt_latent0 = encoder.upsample_latent0(pt_latent0_merged)

    mlx_h0_25 = mlx_h0_reshaped[:25]
    mlx_latent0_merged = mlx_merge(mlx_h0_25, batch_size=1, steps=5, padding=3)
    compare("latent0_merged", pt_latent0_merged, mlx_latent0_merged)

    # Step through upsample_latent0
    sd = model.state_dict()
    w0 = to_mlx_weight_conv(sd["encoder.upsample_latent0.0.weight"])
    mlx_lat0 = mlx_conv2d(mlx_latent0_merged, w0)
    compare("upsample_latent0.0 (Conv1x1)", encoder.upsample_latent0[0](pt_latent0_merged), mlx_lat0)

    w1 = to_mlx_weight_convt(sd["encoder.upsample_latent0.1.weight"])
    mlx_lat0 = mlx_conv_transpose2d(mlx_lat0, w1)
    compare("upsample_latent0.1 (ConvT)", encoder.upsample_latent0[:2](pt_latent0_merged), mlx_lat0)

    w2 = to_mlx_weight_convt(sd["encoder.upsample_latent0.2.weight"])
    mlx_lat0 = mlx_conv_transpose2d(mlx_lat0, w2)
    compare("upsample_latent0.2 (ConvT)", encoder.upsample_latent0[:3](pt_latent0_merged), mlx_lat0)

    w3 = to_mlx_weight_convt(sd["encoder.upsample_latent0.3.weight"])
    mlx_lat0 = mlx_conv_transpose2d(mlx_lat0, w3)
    compare("upsample_latent0 (full)", pt_latent0, mlx_lat0)

    print("\n=== Step 5: decoder first fusion ===")
    # Run full PyTorch encoder to get all 5 features
    with torch.no_grad():
        pt_encodings = encoder(x)

    print("Encoder outputs:")
    for i, enc in enumerate(pt_encodings):
        print(f"  [{i}] shape={list(enc.shape)} range=[{enc.min():.4f}, {enc.max():.4f}]")

    # Run decoder on last encoding
    dec = model.decoder
    pt_proj4 = dec.convs[4](pt_encodings[4])
    mlx_proj4 = to_mlx(pt_proj4)

    # First fusion (no skip)
    pt_fused4 = dec.fusions[4](pt_proj4)
    # MLX: replicate fusion
    fuse_prefix = "decoder.fusions.4"
    # resnet2
    rx = mx.maximum(mlx_proj4, mx.array(0.0))
    rx = mlx_conv2d(rx,
                     to_mlx_weight_conv(sd[f"{fuse_prefix}.resnet2.residual.1.weight"]),
                     mx.array(sd[f"{fuse_prefix}.resnet2.residual.1.bias"].numpy()),
                     padding=(1,1))
    rx = mx.maximum(rx, mx.array(0.0))
    rx = mlx_conv2d(rx,
                     to_mlx_weight_conv(sd[f"{fuse_prefix}.resnet2.residual.3.weight"]),
                     mx.array(sd[f"{fuse_prefix}.resnet2.residual.3.bias"].numpy()),
                     padding=(1,1))
    mlx_after_resnet2 = mlx_proj4 + rx

    deconv_w = to_mlx_weight_convt(sd[f"{fuse_prefix}.deconv.weight"])
    mlx_after_deconv = mlx_conv_transpose2d(mlx_after_resnet2, deconv_w)

    out_conv_w = to_mlx_weight_conv(sd[f"{fuse_prefix}.out_conv.weight"])
    out_conv_b = mx.array(sd[f"{fuse_prefix}.out_conv.bias"].numpy())
    mlx_fused4 = mlx_conv2d(mlx_after_deconv, out_conv_w, out_conv_b)

    compare("decoder fusions[4]", pt_fused4, mlx_fused4)

    print("\n=== Step 6: depth head ===")
    # Run full PyTorch model
    with torch.no_grad():
        pt_depth, pt_fov = model.forward(x)

    print(f"PT depth range: [{pt_depth.min():.4f}, {pt_depth.max():.4f}]")

    print("\nDone! Check where the first large divergence appears.")


if __name__ == "__main__":
    main()
