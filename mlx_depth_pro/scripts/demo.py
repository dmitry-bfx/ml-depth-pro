#!/usr/bin/env python3
"""
Depth Pro Demo: PyTorch ViT (MPS) + MLX C++ post-ViT

Steps:
  1. Load model and image
  2. Run ViT encoders on MPS (patch, image, FOV)
  3. Run post-ViT with MLX C++ lib
  4. Run post-ViT with PyTorch (reference)
  5. Compare outputs
  6. Save depth map
"""

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import ctypes
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, ConvertImageDtype

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT


def load_mlx_lib(path):
    lib = ctypes.CDLL(path)
    lib.depth_pro_mlx_create.restype = ctypes.c_void_p
    lib.depth_pro_mlx_create.argtypes = [ctypes.c_char_p]
    lib.depth_pro_mlx_destroy.restype = None
    lib.depth_pro_mlx_destroy.argtypes = [ctypes.c_void_p]
    lib.depth_pro_mlx_forward.restype = ctypes.c_int
    lib.depth_pro_mlx_forward.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)] * 7
    return lib


def cptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def split_patches(x, overlap_ratio):
    patch_size = 384
    stride = int(patch_size * (1 - overlap_ratio))
    size = x.shape[-1]
    steps = int(math.ceil((size - patch_size) / stride)) + 1
    patches = []
    for j in range(steps):
        for i in range(steps):
            patches.append(x[..., j*stride:j*stride+patch_size, i*stride:i*stride+patch_size])
    return torch.cat(patches, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Depth Pro: PyTorch ViT + MLX post-ViT")
    parser.add_argument("image", help="Input image")
    parser.add_argument("--output", default="depth_output.png")
    parser.add_argument("--checkpoint", default="./checkpoints/depth_pro.pt")
    parser.add_argument("--lib", default="./mlx_depth_pro/build/libdepth_pro_mlx.dylib")
    parser.add_argument("--mlx-weights", default="./mlx_depth_pro/weights/mlx_weights.safetensors")
    args = parser.parse_args()

    # ================================================================
    # 1. Load model and image
    # ================================================================
    print(f"Loading model from {args.checkpoint}")
    t0 = time.perf_counter()
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = args.checkpoint
    model, _ = create_model_and_transforms(config, device=torch.device("cpu"))
    model.eval()
    encoder = model.encoder
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    print(f"Loading MLX lib from {args.lib}")
    lib = load_mlx_lib(args.lib)
    mlx_model = lib.depth_pro_mlx_create(args.mlx_weights.encode())
    if not mlx_model:
        print("  FAILED to load MLX model")
        sys.exit(1)
    print("  Done")

    img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = img.size
    print(f"Image: {args.image} ({orig_w}x{orig_h})")

    transform = Compose([ToTensor(), Normalize([0.5]*3, [0.5]*3), ConvertImageDtype(torch.float32)])
    img_1536 = F.interpolate(transform(img).unsqueeze(0), size=(1536, 1536), mode="bilinear", align_corners=False)

    # ================================================================
    # 2. Run ViT encoders on MPS
    # ================================================================
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"\nViT device: {device}")

    # Move only ViT parts to device
    encoder.patch_encoder.to(device)
    encoder.image_encoder.to(device)
    model.fov.encoder[0].to(device)

    img_dev = img_1536.to(device)

    # Pyramid + patches (cheap, on device)
    x0, x1, x2 = img_dev, \
        F.interpolate(img_dev, scale_factor=0.5, mode="bilinear", align_corners=False), \
        F.interpolate(img_dev, scale_factor=0.25, mode="bilinear", align_corners=False)
    all_patches = torch.cat([split_patches(x0, 0.25), split_patches(x1, 0.5), x2], dim=0)
    fov_input = F.interpolate(img_dev, scale_factor=0.25, mode="bilinear", align_corners=False)

    print(f"Running patch encoder ({all_patches.shape[0]} patches)...", end=" ", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        patch_enc = encoder.patch_encoder(all_patches)
        hook0 = encoder.backbone_highres_hook0.clone()
        hook1 = encoder.backbone_highres_hook1.clone()
    if device.type == "mps":
        torch.mps.synchronize()
    print(f"{time.perf_counter() - t0:.2f}s")

    print(f"Running image encoder...", end=" ", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        image_enc = encoder.image_encoder(x2)
    if device.type == "mps":
        torch.mps.synchronize()
    print(f"{time.perf_counter() - t0:.2f}s")

    print(f"Running FOV encoder...", end=" ", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        fov_vit = model.fov.encoder[0](fov_input)
    if device.type == "mps":
        torch.mps.synchronize()
    print(f"{time.perf_counter() - t0:.2f}s")

    # Transfer to CPU for both backends
    patch_enc = patch_enc.cpu()
    hook0 = hook0.cpu()
    hook1 = hook1.cpu()
    image_enc = image_enc.cpu()
    fov_vit = fov_vit.cpu()

    # ================================================================
    # 3. Run post-ViT with MLX C++
    # ================================================================
    print(f"\nRunning MLX post-ViT...", end=" ", flush=True)

    # Keep numpy arrays alive for the duration of the C call
    pe_np = np.ascontiguousarray(patch_enc.numpy())
    h0_np = np.ascontiguousarray(hook0.numpy())
    h1_np = np.ascontiguousarray(hook1.numpy())
    ie_np = np.ascontiguousarray(image_enc.numpy())
    fv_np = np.ascontiguousarray(fov_vit.numpy())
    mlx_depth = np.zeros((1, 1, 1536, 1536), dtype=np.float32)
    mlx_fov = np.zeros(1, dtype=np.float32)

    t0 = time.perf_counter()
    ret = lib.depth_pro_mlx_forward(
        mlx_model,
        cptr(pe_np), cptr(h0_np), cptr(h1_np), cptr(ie_np), cptr(fv_np),
        cptr(mlx_depth), cptr(mlx_fov),
    )
    mlx_time = time.perf_counter() - t0
    if ret != 0:
        print("FAILED")
        sys.exit(1)
    print(f"{mlx_time:.2f}s")

    # ================================================================
    # 4. Run post-ViT with PyTorch MPS (reference)
    # ================================================================
    print(f"Running PyTorch post-ViT (MPS reference)...", end=" ", flush=True)

    model.to(device)
    pe_dev = patch_enc.to(device)
    h0_dev = hook0.to(device)
    h1_dev = hook1.to(device)
    ie_dev = image_enc.to(device)
    img_dev2 = img_1536.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        pe_ = encoder.reshape_feature(pe_dev, encoder.out_size, encoder.out_size)
        h0_ = encoder.reshape_feature(h0_dev, encoder.out_size, encoder.out_size)
        h1_ = encoder.reshape_feature(h1_dev, encoder.out_size, encoder.out_size)
        ge_ = encoder.reshape_feature(ie_dev, encoder.out_size, encoder.out_size)
        encs = [
            encoder.upsample_latent0(encoder.merge(h0_[:25], 1, 3)),
            encoder.upsample_latent1(encoder.merge(h1_[:25], 1, 3)),
            encoder.upsample0(encoder.merge(pe_[:25], 1, 3)),
            encoder.upsample1(encoder.merge(pe_[25:34], 1, 6)),
            encoder.fuse_lowres(torch.cat((
                encoder.upsample2(pe_[34:]),
                encoder.upsample_lowres(ge_)), dim=1)),
        ]
        features, lowres = model.decoder(encs)
        pt_depth_tensor = model.head(features)
        pt_fov_tensor = model.fov.forward(img_dev2, lowres.detach())
    if device.type == "mps":
        torch.mps.synchronize()
    pt_time = time.perf_counter() - t0
    print(f"{pt_time:.2f}s")

    pt_depth = pt_depth_tensor.cpu().numpy()
    pt_fov_val = float(pt_fov_tensor.cpu().numpy().flat[0])

    # ================================================================
    # 5. Compare
    # ================================================================
    mlx_fov_val = float(mlx_fov[0])
    depth_diff = np.abs(mlx_depth[0, 0] - pt_depth[0, 0])

    print(f"\n{'='*50}")
    print(f"  MLX post-ViT:     {mlx_time*1000:.0f} ms")
    print(f"  PyTorch post-ViT: {pt_time*1000:.0f} ms (MPS)")
    print(f"  Depth max diff:   {depth_diff.max():.6e}")
    print(f"  Depth mean diff:  {depth_diff.mean():.6e}")
    print(f"  FOV MLX:          {mlx_fov_val:.2f} deg")
    print(f"  FOV PyTorch:      {pt_fov_val:.2f} deg")
    print(f"  FOV diff:         {abs(mlx_fov_val - pt_fov_val):.6e} deg")
    match = depth_diff.max() < 0.05  # MPS has ~0.03 precision diff vs CPU/MLX
    print(f"  Match:            {'YES' if match else 'NO'} (tol=0.05, MPS precision)")
    print(f"{'='*50}")

    # ================================================================
    # 6. Save depth map
    # ================================================================
    canonical_inverse_depth = mlx_depth[0, 0]
    f_px = 0.5 * orig_w / np.tan(0.5 * np.deg2rad(mlx_fov_val))
    inverse_depth = canonical_inverse_depth * (orig_w / f_px)

    if orig_h != 1536 or orig_w != 1536:
        inv_t = torch.from_numpy(inverse_depth).unsqueeze(0).unsqueeze(0)
        inv_t = F.interpolate(inv_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        inverse_depth = inv_t.squeeze().numpy()

    depth = 1.0 / np.clip(inverse_depth, 1e-4, 1e4)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    Image.fromarray((depth_norm * 255).astype(np.uint8)).save(args.output)
    print(f"\nDepth range: [{depth.min():.2f}, {depth.max():.2f}] meters")
    print(f"Saved to {args.output}")

    lib.depth_pro_mlx_destroy(mlx_model)


if __name__ == "__main__":
    main()
