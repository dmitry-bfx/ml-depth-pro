#!/usr/bin/env python3
"""Compare MLX C++ lib output vs pure PyTorch reference for numerical correctness."""

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="./build/libdepth_pro_mlx.dylib")
    parser.add_argument("--mlx-weights", default="./mlx_depth_pro/weights/mlx_weights.safetensors")
    parser.add_argument("--checkpoint", default="./checkpoints/depth_pro.pt")
    args = parser.parse_args()

    # --- Load full PyTorch model ---
    print("Loading PyTorch reference model...")
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = args.checkpoint
    model, transform = create_model_and_transforms(config, device=torch.device("cpu"))
    model.eval()

    # --- Create random input ---
    print("Creating test input...")
    torch.manual_seed(42)
    x = torch.randn(1, 3, 1536, 1536)

    # --- PyTorch full forward ---
    print("Running PyTorch reference...")
    with torch.no_grad():
        ref_depth, ref_fov = model.forward(x)
    ref_depth = ref_depth.numpy()
    ref_fov = ref_fov.numpy()

    # --- Extract ViT outputs for MLX ---
    print("Extracting ViT outputs...")
    encoder = model.encoder

    with torch.no_grad():
        x0, x1, x2 = encoder._create_pyramid(x)
        x0_patches = encoder.split(x0, overlap_ratio=0.25)
        x1_patches = encoder.split(x1, overlap_ratio=0.5)
        all_patches = torch.cat([x0_patches, x1_patches, x2], dim=0)

        # Run patch encoder (hooks fire automatically)
        patch_enc_out = encoder.patch_encoder(all_patches)
        hook0_out = encoder.backbone_highres_hook0.clone()
        hook1_out = encoder.backbone_highres_hook1.clone()

        # Image encoder
        image_enc_out = encoder.image_encoder(x2)

        # FOV encoder
        fov_input = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        fov_enc_out = model.fov.encoder(fov_input)  # This includes Linear

        # For the MLX lib, we need just the ViT output before the Linear
        # The FOV encoder is Sequential(vit, Linear). We need the vit output.
        fov_vit_out = model.fov.encoder[0](fov_input)

    # --- Load MLX lib ---
    print("Loading MLX library...")
    lib = ctypes.CDLL(args.lib)
    lib.depth_pro_mlx_create.restype = ctypes.c_void_p
    lib.depth_pro_mlx_create.argtypes = [ctypes.c_char_p]
    lib.depth_pro_mlx_destroy.restype = None
    lib.depth_pro_mlx_destroy.argtypes = [ctypes.c_void_p]
    lib.depth_pro_mlx_forward.restype = ctypes.c_int
    lib.depth_pro_mlx_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    mlx_model = lib.depth_pro_mlx_create(args.mlx_weights.encode())
    assert mlx_model, "Failed to create MLX model"

    def to_cptr(t):
        a = np.ascontiguousarray(t.cpu().float().numpy())
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    mlx_depth = np.zeros((1, 1, 1536, 1536), dtype=np.float32)
    mlx_fov = np.zeros(1, dtype=np.float32)

    ret = lib.depth_pro_mlx_forward(
        mlx_model,
        to_cptr(patch_enc_out),
        to_cptr(hook0_out),
        to_cptr(hook1_out),
        to_cptr(image_enc_out),
        to_cptr(fov_vit_out),
        mlx_depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mlx_fov.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    assert ret == 0, "MLX forward failed"

    lib.depth_pro_mlx_destroy(mlx_model)

    # --- Debug output ranges ---
    print(f"\nRef depth range: [{ref_depth.min():.6f}, {ref_depth.max():.6f}]")
    print(f"MLX depth range: [{mlx_depth.min():.6f}, {mlx_depth.max():.6f}]")
    print(f"Ref FOV: {ref_fov.flat[0]:.6f}")
    print(f"MLX FOV: {mlx_fov[0]:.6f}")
    print(f"Ref depth shape: {ref_depth.shape}, MLX depth shape: {mlx_depth.shape}")
    # Sample specific elements
    for r, c in [(0,0), (100,100), (768,768), (1000,500)]:
        rv = ref_depth[0,0,r,c]
        mv = mlx_depth[0,0,r,c]
        print(f"  [{r},{c}] ref={rv:.6f} mlx={mv:.6f} diff={abs(rv-mv):.6e}")

    # --- Compare ---
    print("\n=== Numerical Comparison ===")
    print(f"ref dtype={ref_depth.dtype} mlx dtype={mlx_depth.dtype}")

    # Direct full-array diff (no slicing)
    full_diff = np.abs(mlx_depth.astype(np.float64) - ref_depth.astype(np.float64))
    print(f"Full array max abs diff: {full_diff.max():.6e}")
    print(f"Full array mean abs diff: {full_diff.mean():.6e}")

    # Relative error
    rel_diff = full_diff / (np.abs(ref_depth.astype(np.float64)) + 1e-8)
    print(f"Full array max rel diff: {rel_diff.max():.6e}")
    print(f"Full array mean rel diff: {rel_diff.mean():.6e}")

    fov_diff = abs(float(mlx_fov[0]) - float(ref_fov.flat[0]))
    print(f"FOV abs error:        {fov_diff:.6e}")

    abs_max = full_diff.max()
    tol = 1e-3
    if abs_max < tol and fov_diff < tol:
        print(f"\nPASS (tolerance {tol})")
    else:
        print(f"\nFAIL (tolerance {tol})")
        sys.exit(1)


if __name__ == "__main__":
    main()
