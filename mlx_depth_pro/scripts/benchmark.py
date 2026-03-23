#!/usr/bin/env python3
"""Benchmark: pure PyTorch vs PyTorch ViT + MLX C++ lib."""

import argparse
import ctypes
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, ConvertImageDtype

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT
from depth_pro.network.vit_factory import VIT_CONFIG_DICT


# ---- Helpers ----

def load_mlx_lib(lib_path):
    lib = ctypes.CDLL(lib_path)
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
    return lib


def cptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def create_pyramid(x):
    x0 = x
    x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
    return x0, x1, x2


def split_patches(x, overlap_ratio):
    patch_size = 384
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = x.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    patches = []
    for j in range(steps):
        j0 = j * patch_stride
        for i in range(steps):
            i0 = i * patch_stride
            patches.append(x[..., j0:j0+patch_size, i0:i0+patch_size])
    return torch.cat(patches, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--lib", default="./mlx_depth_pro/build/libdepth_pro_mlx.dylib")
    parser.add_argument("--mlx-weights", default="./mlx_depth_pro/weights/mlx_weights.safetensors")
    parser.add_argument("--checkpoint", default="./checkpoints/depth_pro.pt")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Load image ----
    img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = img.size
    print(f"Image: {args.image} ({orig_w}x{orig_h})")

    transform = Compose([
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ConvertImageDtype(torch.float32),
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1,3,H,W]
    img_1536 = F.interpolate(img_tensor, size=(1536, 1536), mode="bilinear", align_corners=False)

    # =========================================================
    # PURE PYTORCH
    # =========================================================
    print("\n=== Pure PyTorch ===")
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = args.checkpoint
    model_pt, _ = create_model_and_transforms(config, device=torch.device("cpu"))
    model_pt = model_pt.to(device).eval()
    x_dev = img_1536.to(device)

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            model_pt.forward(x_dev)
        if device.type == "mps":
            torch.mps.synchronize()

    # Timed runs
    pt_times = []
    for i in range(args.runs):
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_depth, pt_fov = model_pt.forward(x_dev)
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        pt_times.append(t1 - t0)
        print(f"  Run {i+1}: {pt_times[-1]*1000:.1f} ms")

    pt_depth_np = pt_depth.cpu().numpy()
    pt_fov_val = float(pt_fov.cpu().numpy().flat[0])
    print(f"  Avg: {np.mean(pt_times)*1000:.1f} ms  (min: {np.min(pt_times)*1000:.1f} ms)")
    print(f"  Depth range: [{pt_depth_np.min():.2f}, {pt_depth_np.max():.2f}]")
    print(f"  FOV: {pt_fov_val:.2f} deg")

    # Free PT model to reclaim memory
    del model_pt
    if device.type == "mps":
        torch.mps.empty_cache()

    # =========================================================
    # PYTORCH VIT + MLX C++ LIB
    # =========================================================
    print("\n=== PyTorch ViT + MLX C++ ===")

    # Load ViTs
    config_vit = VIT_CONFIG_DICT["dinov2l16_384"]
    hook_ids = config_vit.encoder_feature_layer_ids[:2]  # [5, 11]

    # We need the full model to get ViT weights properly split
    # Load from checkpoint and extract the ViT parts
    full_sd = torch.load(args.checkpoint, map_location="cpu")

    from depth_pro.network.vit_factory import create_vit

    patch_encoder = create_vit("dinov2l16_384", use_pretrained=False)
    image_encoder = create_vit("dinov2l16_384", use_pretrained=False)
    fov_encoder = create_vit("dinov2l16_384", use_pretrained=False)

    # Load ViT weights
    def load_vit(model, prefix):
        state = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
        model.load_state_dict(state, strict=False)

    load_vit(patch_encoder, "encoder.patch_encoder.")
    load_vit(image_encoder, "encoder.image_encoder.")
    load_vit(fov_encoder, "fov.encoder.0.")

    # Register hooks on patch encoder
    hook_outputs = {}
    def make_hook(name):
        def fn(mod, inp, out):
            hook_outputs[name] = out
        return fn
    patch_encoder.blocks[hook_ids[0]].register_forward_hook(make_hook("h0"))
    patch_encoder.blocks[hook_ids[1]].register_forward_hook(make_hook("h1"))

    patch_encoder = patch_encoder.to(device).eval()
    image_encoder = image_encoder.to(device).eval()
    fov_encoder = fov_encoder.to(device).eval()

    # Load MLX lib
    lib = load_mlx_lib(args.lib)
    mlx_model = lib.depth_pro_mlx_create(args.mlx_weights.encode())
    assert mlx_model, "Failed to create MLX model"

    # Pre-allocate output buffers
    depth_buf = np.zeros((1, 1, 1536, 1536), dtype=np.float32)
    fov_buf = np.zeros(1, dtype=np.float32)

    def run_hybrid():
        """Run the hybrid PyTorch ViT + MLX pipeline."""
        x_d = img_1536.to(device)

        # PyTorch: pyramid + split + ViT
        x0, x1, x2 = create_pyramid(x_d)
        x0_patches = split_patches(x0, 0.25)
        x1_patches = split_patches(x1, 0.5)
        all_patches = torch.cat([x0_patches, x1_patches, x2], dim=0)

        with torch.no_grad():
            patch_out = patch_encoder(all_patches)
            h0 = hook_outputs["h0"]
            h1 = hook_outputs["h1"]
            img_out = image_encoder(x2)
            fov_input = F.interpolate(x_d, scale_factor=0.25, mode="bilinear", align_corners=False)
            fov_out = fov_encoder(fov_input)

        if device.type == "mps":
            torch.mps.synchronize()

        # Transfer to CPU numpy
        patch_np = patch_out.cpu().float().numpy()
        h0_np = h0.cpu().float().numpy()
        h1_np = h1.cpu().float().numpy()
        img_np = img_out.cpu().float().numpy()
        fov_np = fov_out.cpu().float().numpy()

        # MLX C++ forward
        ret = lib.depth_pro_mlx_forward(
            mlx_model,
            cptr(np.ascontiguousarray(patch_np)),
            cptr(np.ascontiguousarray(h0_np)),
            cptr(np.ascontiguousarray(h1_np)),
            cptr(np.ascontiguousarray(img_np)),
            cptr(np.ascontiguousarray(fov_np)),
            cptr(depth_buf),
            cptr(fov_buf),
        )
        assert ret == 0
        return depth_buf.copy(), float(fov_buf[0])

    # Warmup
    for _ in range(args.warmup):
        run_hybrid()

    # Timed runs — also time ViT vs MLX separately
    hybrid_times = []
    vit_times = []
    mlx_times = []
    for i in range(args.runs):
        x_d = img_1536.to(device)
        x0, x1, x2 = create_pyramid(x_d)
        x0_patches = split_patches(x0, 0.25)
        x1_patches = split_patches(x1, 0.5)
        all_patches = torch.cat([x0_patches, x1_patches, x2], dim=0)
        fov_input = F.interpolate(x_d, scale_factor=0.25, mode="bilinear", align_corners=False)

        if device.type == "mps":
            torch.mps.synchronize()

        # Time ViT portion
        t_start = time.perf_counter()
        with torch.no_grad():
            patch_out = patch_encoder(all_patches)
            h0 = hook_outputs["h0"]
            h1 = hook_outputs["h1"]
            img_out = image_encoder(x2)
            fov_out = fov_encoder(fov_input)
        if device.type == "mps":
            torch.mps.synchronize()
        t_vit = time.perf_counter()

        # Transfer + MLX
        patch_np = np.ascontiguousarray(patch_out.cpu().float().numpy())
        h0_np = np.ascontiguousarray(h0.cpu().float().numpy())
        h1_np = np.ascontiguousarray(h1.cpu().float().numpy())
        img_np = np.ascontiguousarray(img_out.cpu().float().numpy())
        fov_np = np.ascontiguousarray(fov_out.cpu().float().numpy())

        t_transfer = time.perf_counter()

        ret = lib.depth_pro_mlx_forward(
            mlx_model,
            cptr(patch_np), cptr(h0_np), cptr(h1_np),
            cptr(img_np), cptr(fov_np),
            cptr(depth_buf), cptr(fov_buf),
        )
        assert ret == 0
        t_end = time.perf_counter()

        vit_t = t_vit - t_start
        transfer_t = t_transfer - t_vit
        mlx_t = t_end - t_transfer
        total = t_end - t_start

        vit_times.append(vit_t)
        mlx_times.append(mlx_t)
        hybrid_times.append(total)
        print(f"  Run {i+1}: total={total*1000:.1f} ms  "
              f"(ViT={vit_t*1000:.1f} ms, transfer={transfer_t*1000:.1f} ms, MLX={mlx_t*1000:.1f} ms)")

    mlx_depth, mlx_fov = depth_buf.copy(), float(fov_buf[0])
    print(f"  Avg: {np.mean(hybrid_times)*1000:.1f} ms  "
          f"(ViT={np.mean(vit_times)*1000:.1f} ms, MLX={np.mean(mlx_times)*1000:.1f} ms)")
    print(f"  Depth range: [{mlx_depth.min():.2f}, {mlx_depth.max():.2f}]")
    print(f"  FOV: {mlx_fov:.2f} deg")

    # ---- Compare results ----
    print("\n=== Comparison ===")
    depth_diff = np.abs(mlx_depth[0,0] - pt_depth_np[0,0])
    print(f"Depth max abs diff: {depth_diff.max():.6e}")
    print(f"FOV diff: {abs(mlx_fov - pt_fov_val):.6e} deg")
    speedup = np.mean(pt_times) / np.mean(hybrid_times)
    print(f"Speedup: {speedup:.2f}x ({np.mean(pt_times)*1000:.1f} ms → {np.mean(hybrid_times)*1000:.1f} ms)")

    # Save depth visualization
    depth_map = 1.0 / np.clip(mlx_depth[0, 0], 1e-4, 1e4)
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    Image.fromarray((depth_norm * 255).astype(np.uint8)).save("depth_output.png")
    print("Saved depth_output.png")

    lib.depth_pro_mlx_destroy(mlx_model)


if __name__ == "__main__":
    main()
