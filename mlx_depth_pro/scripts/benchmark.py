#!/usr/bin/env python3
"""
1. Verify full pipeline matches: pure PyTorch vs PyTorch ViT + MLX post-ViT
2. Benchmark only the replaced part: PyTorch post-ViT vs MLX C++ post-ViT
"""

import argparse
import ctypes
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


def load_mlx_lib(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.depth_pro_mlx_create.restype = ctypes.c_void_p
    lib.depth_pro_mlx_create.argtypes = [ctypes.c_char_p]
    lib.depth_pro_mlx_destroy.restype = None
    lib.depth_pro_mlx_destroy.argtypes = [ctypes.c_void_p]
    lib.depth_pro_mlx_forward.restype = ctypes.c_int
    lib.depth_pro_mlx_forward.argtypes = [
        ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)] * 7
    return lib


def cptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--lib", default="./mlx_depth_pro/build/libdepth_pro_mlx.dylib")
    parser.add_argument("--mlx-weights", default="./mlx_depth_pro/weights/mlx_weights.safetensors")
    parser.add_argument("--checkpoint", default="./checkpoints/depth_pro.pt")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image} ({img.size[0]}x{img.size[1]})")

    transform = Compose([ToTensor(), Normalize([0.5]*3, [0.5]*3), ConvertImageDtype(torch.float32)])
    img_1536 = F.interpolate(transform(img).unsqueeze(0), size=(1536, 1536), mode="bilinear", align_corners=False)

    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = args.checkpoint
    model, _ = create_model_and_transforms(config, device=torch.device("cpu"))
    model.eval()
    encoder = model.encoder

    # --- Full PyTorch pipeline (reference) ---
    print("\nRunning full PyTorch pipeline...")
    with torch.no_grad():
        ref_depth, ref_fov = model.forward(img_1536)
    ref_depth_np = ref_depth.numpy()
    ref_fov_val = float(ref_fov.numpy().flat[0])
    print(f"  Depth: [{ref_depth_np.min():.4f}, {ref_depth_np.max():.4f}], FOV: {ref_fov_val:.2f} deg")

    # --- Extract ViT outputs ---
    with torch.no_grad():
        x0, x1, x2 = encoder._create_pyramid(img_1536)
        patches = torch.cat([encoder.split(x0, 0.25), encoder.split(x1, 0.5), x2], dim=0)
        patch_enc = encoder.patch_encoder(patches)
        hook0 = encoder.backbone_highres_hook0.clone()
        hook1 = encoder.backbone_highres_hook1.clone()
        image_enc = encoder.image_encoder(x2)
        fov_vit = model.fov.encoder[0](
            F.interpolate(img_1536, scale_factor=0.25, mode="bilinear", align_corners=False))

    # --- MLX pipeline ---
    lib = load_mlx_lib(args.lib)
    mlx_model = lib.depth_pro_mlx_create(args.mlx_weights.encode())
    assert mlx_model

    pe = np.ascontiguousarray(patch_enc.numpy())
    h0 = np.ascontiguousarray(hook0.numpy())
    h1 = np.ascontiguousarray(hook1.numpy())
    ie = np.ascontiguousarray(image_enc.numpy())
    fe = np.ascontiguousarray(fov_vit.numpy())
    depth_buf = np.zeros((1, 1, 1536, 1536), dtype=np.float32)
    fov_buf = np.zeros(1, dtype=np.float32)

    lib.depth_pro_mlx_forward(mlx_model, cptr(pe), cptr(h0), cptr(h1), cptr(ie), cptr(fe), cptr(depth_buf), cptr(fov_buf))

    # --- 1. Verify outputs match ---
    print("\n=== Pipeline Match ===")
    diff = np.abs(depth_buf[0, 0] - ref_depth_np[0, 0])
    fov_diff = abs(float(fov_buf[0]) - ref_fov_val)
    print(f"Depth max diff: {diff.max():.6e}  mean: {diff.mean():.6e}")
    print(f"FOV diff: {fov_diff:.6e} deg")
    print(f"{'PASS' if diff.max() < 0.1 and fov_diff < 0.1 else 'FAIL'}")

    # --- 2. Benchmark replaced part ---
    print(f"\n=== Benchmark post-ViT ({args.runs} runs) ===")

    def make_pt_post_vit(dev):
        """Create post-ViT runner on given device."""
        enc_d = encoder.to(dev)
        dec_d = model.decoder.to(dev)
        head_d = model.head.to(dev)
        fov_d = model.fov.to(dev)
        pe_d = patch_enc.to(dev)
        h0_d = hook0.to(dev)
        h1_d = hook1.to(dev)
        ie_d = image_enc.to(dev)
        img_d = img_1536.to(dev)
        sync = torch.mps.synchronize if dev.type == "mps" else lambda: None

        def run():
            with torch.no_grad():
                pe_ = enc_d.reshape_feature(pe_d, enc_d.out_size, enc_d.out_size)
                h0_ = enc_d.reshape_feature(h0_d, enc_d.out_size, enc_d.out_size)
                h1_ = enc_d.reshape_feature(h1_d, enc_d.out_size, enc_d.out_size)
                ge_ = enc_d.reshape_feature(ie_d, enc_d.out_size, enc_d.out_size)
                encs = [
                    enc_d.upsample_latent0(enc_d.merge(h0_[:25], 1, 3)),
                    enc_d.upsample_latent1(enc_d.merge(h1_[:25], 1, 3)),
                    enc_d.upsample0(enc_d.merge(pe_[:25], 1, 3)),
                    enc_d.upsample1(enc_d.merge(pe_[25:34], 1, 6)),
                    enc_d.fuse_lowres(torch.cat((
                        enc_d.upsample2(pe_[34:]),
                        enc_d.upsample_lowres(ge_)), dim=1)),
                ]
                features, lowres = dec_d(encs)
                head_d(features)
                fov_d.forward(img_d, lowres.detach())
            sync()
        return run

    def mlx_post_vit():
        lib.depth_pro_mlx_forward(mlx_model, cptr(pe), cptr(h0), cptr(h1), cptr(ie), cptr(fe), cptr(depth_buf), cptr(fov_buf))

    def bench(name, fn):
        for _ in range(2):
            fn()
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        avg = np.mean(times) * 1000
        print(f"{name:20s} {avg:8.1f} ms avg  ({', '.join(f'{t*1000:.1f}' for t in times)})")
        return avg

    pt_mps = bench("PyTorch MPS", make_pt_post_vit(torch.device("mps")))
    encoder.to("cpu"); model.decoder.to("cpu"); model.head.to("cpu"); model.fov.to("cpu")

    mlx_avg = bench("MLX C++", mlx_post_vit)

    print(f"\n--- Speedup ---")
    print(f"MLX vs PyTorch MPS: {pt_mps/mlx_avg:.2f}x")

    lib.depth_pro_mlx_destroy(mlx_model)


if __name__ == "__main__":
    main()
