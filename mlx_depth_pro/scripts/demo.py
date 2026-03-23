#!/usr/bin/env python3
"""Depth Pro demo: PyTorch ViT + MLX C++ lib for post-ViT processing."""

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, ConvertImageDtype

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth_pro.network.vit_factory import create_vit, VIT_CONFIG_DICT


class ViTWithHooks:
    """Wrapper to run the patch encoder and capture intermediate hook outputs."""

    def __init__(self, model: nn.Module, hook_block_ids: list[int]):
        self.model = model
        self.hook_outputs = {}

        for idx in hook_block_ids:
            self.model.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )

    def _make_hook(self, idx):
        def hook_fn(module, input, output):
            self.hook_outputs[idx] = output
        return hook_fn

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.hook_outputs.clear()
        return self.model(x)


def create_pyramid(x: torch.Tensor):
    """Create 3-level image pyramid from 1536x1536 input."""
    x0 = x
    x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
    return x0, x1, x2


def split_patches(x: torch.Tensor, overlap_ratio: float) -> torch.Tensor:
    """Split image into overlapping 384x384 patches with sliding window."""
    import math

    patch_size = 384
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = x.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

    patches = []
    for j in range(steps):
        j0 = j * patch_stride
        j1 = j0 + patch_size
        for i in range(steps):
            i0 = i * patch_stride
            i1 = i0 + patch_size
            patches.append(x[..., j0:j1, i0:i1])

    return torch.cat(patches, dim=0)


def load_mlx_lib(lib_path: str):
    """Load the MLX C++ shared library."""
    lib = ctypes.CDLL(lib_path)

    # depth_pro_mlx_create
    lib.depth_pro_mlx_create.restype = ctypes.c_void_p
    lib.depth_pro_mlx_create.argtypes = [ctypes.c_char_p]

    # depth_pro_mlx_destroy
    lib.depth_pro_mlx_destroy.restype = None
    lib.depth_pro_mlx_destroy.argtypes = [ctypes.c_void_p]

    # depth_pro_mlx_forward
    lib.depth_pro_mlx_forward.restype = ctypes.c_int
    lib.depth_pro_mlx_forward.argtypes = [
        ctypes.c_void_p,                         # model
        ctypes.POINTER(ctypes.c_float),          # patch_enc_out
        ctypes.POINTER(ctypes.c_float),          # hook0_out
        ctypes.POINTER(ctypes.c_float),          # hook1_out
        ctypes.POINTER(ctypes.c_float),          # image_enc_out
        ctypes.POINTER(ctypes.c_float),          # fov_enc_out
        ctypes.POINTER(ctypes.c_float),          # depth_out
        ctypes.POINTER(ctypes.c_float),          # fov_deg_out
    ]

    return lib


def numpy_to_cptr(arr: np.ndarray):
    """Get a ctypes float pointer from a contiguous numpy array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def main():
    parser = argparse.ArgumentParser(description="Depth Pro demo with MLX backend")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--lib", default="./build/libdepth_pro_mlx.dylib",
                        help="Path to MLX shared library")
    parser.add_argument("--mlx-weights", default="./mlx_depth_pro/weights/mlx_weights.safetensors",
                        help="Path to MLX safetensors weights")
    parser.add_argument("--vit-weights", default="./mlx_depth_pro/weights/vit_weights.pt",
                        help="Path to ViT PyTorch weights")
    parser.add_argument("--output", default="depth_output.png",
                        help="Output depth map path")
    parser.add_argument("--device", default="mps",
                        help="PyTorch device for ViT (mps, cpu, cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() or args.device != "mps" else "cpu")
    print(f"Using device: {device}")

    # --- Load ViT models ---
    print("Loading ViT models...")
    config = VIT_CONFIG_DICT["dinov2l16_384"]
    hook_block_ids = config.encoder_feature_layer_ids  # [5, 11, 17, 23]

    patch_encoder = create_vit("dinov2l16_384", use_pretrained=False)
    image_encoder = create_vit("dinov2l16_384", use_pretrained=False)
    fov_encoder = create_vit("dinov2l16_384", use_pretrained=False)

    # Load ViT weights
    vit_state = torch.load(args.vit_weights, map_location="cpu")

    def load_vit_weights(model, prefix):
        state = {}
        for k, v in vit_state.items():
            if k.startswith(prefix):
                # Remove prefix to get model-relative key
                new_key = k[len(prefix):]
                state[new_key] = v
        missing, unexpected = model.load_state_dict(state, strict=False)
        # fc_norm is only for classification head, safe to ignore
        missing = [k for k in missing if "fc_norm" not in k]
        if missing:
            print(f"  Warning: missing keys for {prefix}: {missing[:5]}...")

    load_vit_weights(patch_encoder, "encoder.patch_encoder.")
    load_vit_weights(image_encoder, "encoder.image_encoder.")
    load_vit_weights(fov_encoder, "fov.encoder.0.")

    patch_encoder_hooks = ViTWithHooks(patch_encoder, hook_block_ids[:2])  # hooks on [5, 11]

    patch_encoder.to(device).eval()
    image_encoder.to(device).eval()
    fov_encoder.to(device).eval()

    # --- Load MLX lib ---
    print("Loading MLX library...")
    lib = load_mlx_lib(args.lib)
    mlx_model = lib.depth_pro_mlx_create(args.mlx_weights.encode())
    if not mlx_model:
        print("Failed to create MLX model")
        sys.exit(1)

    # --- Preprocess image ---
    print("Processing image...")
    img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = img.size

    transform = Compose([
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ConvertImageDtype(torch.float32),
    ])

    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

    # Resize to 1536x1536 for the network
    img_1536 = F.interpolate(
        img_tensor, size=(1536, 1536), mode="bilinear", align_corners=False
    ).to(device)

    # --- PyTorch: create pyramid and split ---
    x0, x1, x2 = create_pyramid(img_1536)

    x0_patches = split_patches(x0, overlap_ratio=0.25)  # [25, 3, 384, 384]
    x1_patches = split_patches(x1, overlap_ratio=0.5)   # [9, 3, 384, 384]
    x2_patches = x2                                       # [1, 3, 384, 384]

    all_patches = torch.cat([x0_patches, x1_patches, x2_patches], dim=0)  # [35, 3, 384, 384]

    # --- PyTorch: run ViTs ---
    print("Running ViT encoders...")
    with torch.no_grad():
        # Patch encoder with hooks
        patch_enc_out = patch_encoder_hooks(all_patches)  # [35, 577, 1024]
        hook0_out = patch_encoder_hooks.hook_outputs[hook_block_ids[0]]  # [35, 577, 1024]
        hook1_out = patch_encoder_hooks.hook_outputs[hook_block_ids[1]]  # [35, 577, 1024]

        # Image encoder
        image_enc_out = image_encoder(x2_patches)  # [1, 577, 1024]

        # FOV encoder
        fov_input = F.interpolate(img_1536, scale_factor=0.25, mode="bilinear", align_corners=False)
        fov_enc_out = fov_encoder(fov_input)  # [1, 577, 1024]

    # Move to CPU numpy
    patch_enc_np = patch_enc_out.cpu().float().numpy()
    hook0_np = hook0_out.cpu().float().numpy()
    hook1_np = hook1_out.cpu().float().numpy()
    image_enc_np = image_enc_out.cpu().float().numpy()
    fov_enc_np = fov_enc_out.cpu().float().numpy()

    # --- MLX: run post-ViT processing ---
    print("Running MLX post-ViT processing...")
    depth_out = np.zeros((1, 1, 1536, 1536), dtype=np.float32)
    fov_deg_out = np.zeros(1, dtype=np.float32)

    ret = lib.depth_pro_mlx_forward(
        mlx_model,
        numpy_to_cptr(patch_enc_np),
        numpy_to_cptr(hook0_np),
        numpy_to_cptr(hook1_np),
        numpy_to_cptr(image_enc_np),
        numpy_to_cptr(fov_enc_np),
        numpy_to_cptr(depth_out),
        fov_deg_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    if ret != 0:
        print("MLX forward failed")
        sys.exit(1)

    # --- Post-process ---
    canonical_inverse_depth = depth_out[0, 0]  # [1536, 1536]
    fov_deg = float(fov_deg_out[0])

    print(f"Estimated FOV: {fov_deg:.1f} degrees")

    # Compute metric depth
    f_px = 0.5 * orig_w / np.tan(0.5 * np.deg2rad(fov_deg))
    inverse_depth = canonical_inverse_depth * (orig_w / f_px)

    # Resize to original resolution if needed
    if orig_h != 1536 or orig_w != 1536:
        inverse_depth_t = torch.from_numpy(inverse_depth).unsqueeze(0).unsqueeze(0)
        inverse_depth_t = F.interpolate(
            inverse_depth_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        inverse_depth = inverse_depth_t.squeeze().numpy()

    depth = 1.0 / np.clip(inverse_depth, 1e-4, 1e4)

    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")

    # Save as normalized visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_img = Image.fromarray((depth_normalized * 255).astype(np.uint8))
    depth_img.save(args.output)
    print(f"Saved depth map to {args.output}")

    # Cleanup
    lib.depth_pro_mlx_destroy(mlx_model)


if __name__ == "__main__":
    main()
