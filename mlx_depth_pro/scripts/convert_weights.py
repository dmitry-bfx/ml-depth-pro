#!/usr/bin/env python3
"""Split Depth Pro weights into ViT (PyTorch) and non-ViT (safetensors for MLX C++)."""

import argparse
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file
except ImportError:
    print("Install safetensors: pip install safetensors")
    sys.exit(1)


# Prefixes that belong to the ViT backbones (stay in PyTorch)
VIT_PREFIXES = (
    "encoder.patch_encoder.",
    "encoder.image_encoder.",
    "fov.encoder.0.",  # The ViT inside FOV's nn.Sequential([fov_encoder, Linear])
)


def is_vit_key(key: str) -> bool:
    return any(key.startswith(p) for p in VIT_PREFIXES)


# ConvTranspose2d weight keys — these use [in, out, kH, kW] in PyTorch
# and need permute(1,2,3,0) to get MLX's [out, kH, kW, in]
CONV_TRANSPOSE_PATTERNS = (
    "deconv.weight",           # decoder.fusions.*.deconv.weight
    "upsample_lowres.weight",  # encoder.upsample_lowres.weight
    "head.1.weight",           # depth head ConvTranspose2d
)


def is_conv_transpose_key(key: str) -> bool:
    """Check if a key belongs to a ConvTranspose2d layer."""
    if any(p in key for p in CONV_TRANSPOSE_PATTERNS):
        return True
    # encoder.upsample*.{1,2,3}.weight are ConvTranspose2d
    # encoder.upsample*.0.weight are Conv2d (1x1 projection)
    import re
    m = re.match(r"encoder\.upsample\w+\.([1-9]\d*)\.weight", key)
    return m is not None


def convert(checkpoint_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    vit_weights = {}
    mlx_weights = {}

    for key, tensor in state_dict.items():
        if is_vit_key(key):
            vit_weights[key] = tensor
        else:
            if tensor.ndim == 4:
                if is_conv_transpose_key(key):
                    # ConvTranspose2d: PyTorch [in, out, kH, kW] -> MLX [out, kH, kW, in]
                    tensor = tensor.permute(1, 2, 3, 0).contiguous()
                else:
                    # Conv2d: PyTorch [out, in, kH, kW] -> MLX [out, kH, kW, in]
                    tensor = tensor.permute(0, 2, 3, 1).contiguous()
            mlx_weights[key] = tensor

    print(f"ViT keys: {len(vit_weights)}")
    print(f"MLX keys: {len(mlx_weights)}")

    # Save ViT weights as regular PyTorch checkpoint
    vit_path = out / "vit_weights.pt"
    torch.save(vit_weights, vit_path)
    print(f"Saved ViT weights to {vit_path}")

    # Save MLX weights as safetensors (MLX C++ loads these natively)
    # Convert all to float32 contiguous for safetensors
    mlx_weights_f32 = {k: v.float().contiguous() for k, v in mlx_weights.items()}
    mlx_path = out / "mlx_weights.safetensors"
    save_file(mlx_weights_f32, str(mlx_path))
    print(f"Saved MLX weights to {mlx_path}")

    # Print key summary
    print("\n--- MLX weight keys ---")
    for key in sorted(mlx_weights.keys()):
        shape = list(mlx_weights[key].shape)
        print(f"  {key}: {shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/depth_pro.pt",
        help="Path to depth_pro.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="./mlx_depth_pro/weights",
        help="Output directory for split weights",
    )
    args = parser.parse_args()
    convert(args.checkpoint, args.output_dir)
