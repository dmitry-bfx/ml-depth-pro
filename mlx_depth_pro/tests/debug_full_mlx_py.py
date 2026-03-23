#!/usr/bin/env python3
"""Full pipeline in MLX Python using safetensors weights — compare with PyTorch."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT


def load_safetensors_weights(path):
    """Load safetensors and return dict of MLX arrays."""
    weights, _ = mx.load(path, return_metadata=True)
    return weights


def mlx_conv2d(x, w, b=None, stride=(1,1), padding=(0,0)):
    y = mx.conv2d(x, w, stride=stride, padding=padding)
    if b is not None:
        y = y + b
    return y

def mlx_convt2d(x, w, b=None, stride=(2,2)):
    y = mx.conv_transpose2d(x, w, stride=stride)
    if b is not None:
        y = y + b
    return y

def mlx_relu(x):
    return mx.maximum(x, mx.array(0.0))

def mlx_reshape_feature(emb, h=24, w=24):
    n, _, c = emb.shape
    tokens = emb[:, 1:, :]  # remove CLS
    return mx.reshape(tokens, [n, h, w, c])

def mlx_merge(x, batch_size, steps, padding):
    idx = 0
    rows = []
    for j in range(steps):
        row = []
        for i in range(steps):
            p = x[batch_size * idx : batch_size * (idx + 1)]
            hs, he = 0, p.shape[1]
            ws, we = 0, p.shape[2]
            if j != 0: hs = padding
            if j != steps - 1: he -= padding
            if i != 0: ws = padding
            if i != steps - 1: we -= padding
            row.append(p[:, hs:he, ws:we, :])
            idx += 1
        rows.append(mx.concatenate(row, axis=2))
    return mx.concatenate(rows, axis=1)


def w(weights, key):
    if key not in weights:
        raise KeyError(f"Missing weight: {key}")
    return weights[key]


def run_upsample_block(x, weights, prefix, num_convt):
    x = mlx_conv2d(x, w(weights, prefix + ".0.weight"))
    for i in range(num_convt):
        x = mlx_convt2d(x, w(weights, prefix + "." + str(i+1) + ".weight"))
    return x


def residual_block(x, w1, b1, w2, b2):
    r = mlx_relu(x)
    r = mlx_conv2d(r, w1, b1, padding=(1,1))
    r = mlx_relu(r)
    r = mlx_conv2d(r, w2, b2, padding=(1,1))
    return x + r


def fusion_block(x0, x1, weights, prefix, use_deconv):
    x = x0
    if x1 is not None:
        res = residual_block(
            x1,
            w(weights, prefix + ".resnet1.residual.1.weight"),
            w(weights, prefix + ".resnet1.residual.1.bias"),
            w(weights, prefix + ".resnet1.residual.3.weight"),
            w(weights, prefix + ".resnet1.residual.3.bias"))
        x = x + res
    x = residual_block(
        x,
        w(weights, prefix + ".resnet2.residual.1.weight"),
        w(weights, prefix + ".resnet2.residual.1.bias"),
        w(weights, prefix + ".resnet2.residual.3.weight"),
        w(weights, prefix + ".resnet2.residual.3.bias"))
    if use_deconv:
        x = mlx_convt2d(x, w(weights, prefix + ".deconv.weight"))
    x = mlx_conv2d(x, w(weights, prefix + ".out_conv.weight"),
                    w(weights, prefix + ".out_conv.bias"))
    return x


def main():
    USE_SAFETENSORS = "--safetensors" in sys.argv
    if USE_SAFETENSORS:
        print("Loading weights from safetensors...")
        weights = load_safetensors_weights("./mlx_depth_pro/weights/mlx_weights.safetensors")
    else:
        print("Loading weights from PyTorch state dict (direct conversion)...")
        sd_raw = torch.load("./checkpoints/depth_pro.pt", map_location="cpu")
        VIT_PREFIXES = ("encoder.patch_encoder.", "encoder.image_encoder.", "fov.encoder.0.")
        weights = {}
        # Import the converter's key classification
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from convert_weights import is_conv_transpose_key
        for k, v in sd_raw.items():
            if any(k.startswith(p) for p in VIT_PREFIXES):
                continue
            t = v.float()
            if t.ndim == 4:
                if is_conv_transpose_key(k):
                    t = t.permute(1, 2, 3, 0).contiguous()  # IOHW -> OHWI
                else:
                    t = t.permute(0, 2, 3, 1).contiguous()  # OIHW -> OHWI
            weights[k] = mx.array(t.numpy())
    print(f"Loaded {len(weights)} weight keys")

    print("\nLoading PyTorch model...")
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = "./checkpoints/depth_pro.pt"
    model, _ = create_model_and_transforms(config, device=torch.device("cpu"))
    model.eval()
    encoder = model.encoder

    torch.manual_seed(42)
    x = torch.randn(1, 3, 1536, 1536)

    # Run PyTorch
    print("Running PyTorch reference...")
    with torch.no_grad():
        pt_depth, pt_fov = model.forward(x)

        # Extract ViT outputs
        x0, x1, x2 = encoder._create_pyramid(x)
        x0_patches = encoder.split(x0, overlap_ratio=0.25)
        x1_patches = encoder.split(x1, overlap_ratio=0.5)
        all_patches = torch.cat([x0_patches, x1_patches, x2], dim=0)
        patch_enc = encoder.patch_encoder(all_patches)
        hook0 = encoder.backbone_highres_hook0.clone()
        hook1 = encoder.backbone_highres_hook1.clone()
        image_enc = encoder.image_encoder(x2)
        fov_input = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        fov_vit = model.fov.encoder[0](fov_input)

    # Convert ViT outputs to MLX
    m_patch = mx.array(patch_enc.numpy())
    m_hook0 = mx.array(hook0.numpy())
    m_hook1 = mx.array(hook1.numpy())
    m_image = mx.array(image_enc.numpy())
    m_fov_vit = mx.array(fov_vit.numpy())

    # ===== Encoder post-ViT in MLX =====
    print("\nRunning MLX encoder post-ViT...")
    x_pyr = mlx_reshape_feature(m_patch)
    x_h0 = mlx_reshape_feature(m_hook0)
    x_h1 = mlx_reshape_feature(m_hook1)
    x_glob = mlx_reshape_feature(m_image)

    lat0 = mlx_merge(x_h0[:25], 1, 5, 3)
    lat1 = mlx_merge(x_h1[:25], 1, 5, 3)
    x0_feat = mlx_merge(x_pyr[:25], 1, 5, 3)
    x1_feat = mlx_merge(x_pyr[25:34], 1, 3, 6)
    x2_feat = x_pyr[34:]

    lat0 = run_upsample_block(lat0, weights, "encoder.upsample_latent0", 3)
    lat1 = run_upsample_block(lat1, weights, "encoder.upsample_latent1", 2)
    x0_feat = run_upsample_block(x0_feat, weights, "encoder.upsample0", 1)
    x1_feat = run_upsample_block(x1_feat, weights, "encoder.upsample1", 1)
    x2_feat = run_upsample_block(x2_feat, weights, "encoder.upsample2", 1)

    x_glob = mlx_convt2d(x_glob,
                          w(weights, "encoder.upsample_lowres.weight"),
                          w(weights, "encoder.upsample_lowres.bias"))
    fused_glob = mlx_conv2d(
        mx.concatenate([x2_feat, x_glob], axis=3),
        w(weights, "encoder.fuse_lowres.weight"),
        w(weights, "encoder.fuse_lowres.bias"))

    encodings = [lat0, lat1, x0_feat, x1_feat, fused_glob]

    # Compare encoder outputs with PyTorch
    # Use the SAME ViT outputs (not a separate encoder(x) call) to eliminate ViT non-determinism
    print("\nComparing encoder outputs (using same ViT outputs)...")
    with torch.no_grad():
        # Process the extracted ViT outputs through PyTorch encoder manually
        pt_pyr = encoder.reshape_feature(patch_enc, encoder.out_size, encoder.out_size)
        pt_h0 = encoder.reshape_feature(hook0, encoder.out_size, encoder.out_size)
        pt_h1 = encoder.reshape_feature(hook1, encoder.out_size, encoder.out_size)
        pt_glob = encoder.reshape_feature(image_enc, encoder.out_size, encoder.out_size)

        pt_lat0 = encoder.upsample_latent0(encoder.merge(pt_h0[:25], 1, 3))
        pt_lat1 = encoder.upsample_latent1(encoder.merge(pt_h1[:25], 1, 3))
        pt_x0 = encoder.upsample0(encoder.merge(pt_pyr[:25], 1, 3))
        pt_x1 = encoder.upsample1(encoder.merge(pt_pyr[25:34], 1, 6))
        pt_x2 = encoder.upsample2(pt_pyr[34:])
        pt_glob_up = encoder.upsample_lowres(pt_glob)
        pt_fused = encoder.fuse_lowres(torch.cat((pt_x2, pt_glob_up), dim=1))

    pt_encodings_manual = [pt_lat0, pt_lat1, pt_x0, pt_x1, pt_fused]
    for i in range(5):
        mx.eval(encodings[i])
        pt_np = pt_encodings_manual[i].numpy().transpose(0, 2, 3, 1)
        mlx_np = np.array(encodings[i])
        diff = np.abs(pt_np - mlx_np).max()
        print(f"  enc[{i}] pt_shape={list(pt_encodings_manual[i].shape)} mlx_shape={list(encodings[i].shape)} max_diff={diff:.6e}")

    # Compare ALL safetensors weights vs direct PyTorch conversion
    sd = model.state_dict()
    print("\nWeight comparison (safetensors vs PyTorch):")
    mismatches = 0
    for key in sorted(weights.keys()):
        st_w = np.array(w(weights, key))
        if key in sd:
            pt_tensor = sd[key]
            if pt_tensor.ndim == 4:
                pt_w = pt_tensor.permute(0, 2, 3, 1).contiguous().numpy()
            else:
                pt_w = pt_tensor.numpy()
            match = np.allclose(pt_w, st_w, atol=1e-6)
            if not match:
                diff = np.abs(pt_w - st_w).max()
                print(f"  MISMATCH {key}: shapes pt={pt_w.shape} st={st_w.shape} max_diff={diff:.6e}")
                mismatches += 1
        else:
            print(f"  MISSING in PT state dict: {key}")
            mismatches += 1
    print(f"  {mismatches} mismatches out of {len(weights)} keys")

    # ===== Decoder in MLX =====
    print("\nRunning MLX decoder...")
    projected = []
    for i in range(5):
        key = f"decoder.convs.{i}.weight"
        if key in weights:
            pad = (0,0) if i == 0 else (1,1)
            projected.append(mlx_conv2d(encodings[i], w(weights, key), padding=pad))
        else:
            projected.append(encodings[i])

    features = projected[4]
    lowres_features = features

    features = fusion_block(features, None, weights, "decoder.fusions.4", True)

    # Compare after first fusion
    with torch.no_grad():
        pt_dec_features, pt_lowres = model.decoder(pt_encodings_manual)
    mx.eval(features)
    # At this point features = after fusions[4], shape [1, 96, 96, 256]
    print(f"\nAfter fusions[4]: mlx range [{np.array(features).min():.4f}, {np.array(features).max():.4f}]")

    for i in range(3, -1, -1):
        features = fusion_block(features, projected[i], weights,
                                f"decoder.fusions.{i}", i != 0)
        mx.eval(features)
        print(f"After fusions[{i}]: mlx range [{np.array(features).min():.4f}, {np.array(features).max():.4f}]")

    # Run PT decoder on same encoder outputs
    with torch.no_grad():
        pt_dec_features, pt_lowres = model.decoder(pt_encodings_manual)
    pt_dec_np = pt_dec_features.numpy().transpose(0, 2, 3, 1)
    mlx_dec_np = np.array(features)
    dec_diff = np.abs(pt_dec_np - mlx_dec_np).max()
    print(f"\nDecoder output diff: {dec_diff:.6e}")
    print(f"  PT range:  [{pt_dec_np.min():.4f}, {pt_dec_np.max():.4f}]")
    print(f"  MLX range: [{mlx_dec_np.min():.4f}, {mlx_dec_np.max():.4f}]")

    # ===== Head in MLX =====
    print("Running MLX head...")
    hx = mlx_conv2d(features, w(weights, "head.0.weight"), w(weights, "head.0.bias"),
                     padding=(1,1))
    hx = mlx_convt2d(hx, w(weights, "head.1.weight"), w(weights, "head.1.bias"))
    hx = mlx_conv2d(hx, w(weights, "head.2.weight"), w(weights, "head.2.bias"),
                     padding=(1,1))
    hx = mlx_relu(hx)
    hx = mlx_conv2d(hx, w(weights, "head.4.weight"), w(weights, "head.4.bias"))
    hx = mlx_relu(hx)

    mx.eval(hx)

    # ===== FOV in MLX =====
    print("Running MLX FOV head...")

    # Debug: compare linear output
    with torch.no_grad():
        pt_fov_linear_out = model.fov.encoder(fov_input)  # Sequential(ViT, Linear) → [1,577,128]
        pt_fov_tokens = pt_fov_linear_out[:, 1:]  # [1, 576, 128]
        pt_fov_permuted = pt_fov_tokens.permute(0, 2, 1)  # [1, 128, 576]
        pt_fov_reshaped = pt_fov_permuted.reshape(1, 128, 24, 24)  # [1, 128, 24, 24]

    fov_tokens = m_fov_vit[:, 1:, :]  # [1, 576, 1024]
    fov_linear_w = w(weights, "fov.encoder.1.weight")
    fov_linear_b = w(weights, "fov.encoder.1.bias")
    fov_x = fov_tokens @ mx.transpose(fov_linear_w) + fov_linear_b  # [1, 576, 128]
    mx.eval(fov_x)

    # Compare linear output
    pt_lin_nhwc = pt_fov_tokens.numpy()  # [1, 576, 128]
    mlx_lin = np.array(fov_x)  # [1, 576, 128]
    print(f"  FOV linear: pt_range=[{pt_lin_nhwc.min():.4f},{pt_lin_nhwc.max():.4f}] "
          f"mlx_range=[{mlx_lin.min():.4f},{mlx_lin.max():.4f}] "
          f"max_diff={np.abs(pt_lin_nhwc - mlx_lin).max():.6e}")

    fov_x = mx.reshape(fov_x, [1, 24, 24, 128])

    # Compare reshape: PT [1,128,24,24] NCHW vs MLX [1,24,24,128] NHWC
    pt_reshaped_nhwc = pt_fov_reshaped.numpy().transpose(0, 2, 3, 1)
    mlx_reshaped = np.array(fov_x)
    print(f"  FOV reshape: max_diff={np.abs(pt_reshaped_nhwc - mlx_reshaped).max():.6e}")

    # Compare lowres input
    mx.eval(lowres_features)
    pt_lowres_nhwc = pt_lowres.numpy().transpose(0, 2, 3, 1)
    mlx_lowres_np = np.array(lowres_features)
    print(f"  FOV lowres input: max_diff={np.abs(pt_lowres_nhwc - mlx_lowres_np).max():.6e}")
    print(f"    PT lowres range: [{pt_lowres_nhwc.min():.4f}, {pt_lowres_nhwc.max():.4f}] shape={pt_lowres_nhwc.shape}")
    print(f"    MLX lowres range: [{mlx_lowres_np.min():.4f}, {mlx_lowres_np.max():.4f}] shape={mlx_lowres_np.shape}")

    fov_low = mlx_conv2d(lowres_features,
                          w(weights, "fov.downsample.0.weight"),
                          w(weights, "fov.downsample.0.bias"),
                          stride=(2,2), padding=(1,1))
    fov_low = mlx_relu(fov_low)  # downsample is Sequential(Conv2d, ReLU)

    # Compare downsample output
    with torch.no_grad():
        pt_fov_down = model.fov.downsample(pt_lowres)
    mx.eval(fov_low)
    pt_down_nhwc = pt_fov_down.numpy().transpose(0, 2, 3, 1)
    mlx_down_np = np.array(fov_low)
    print(f"  FOV downsample: max_diff={np.abs(pt_down_nhwc - mlx_down_np).max():.6e}")

    fov_x = fov_x + fov_low

    fov_x = mlx_conv2d(fov_x, w(weights, "fov.head.0.weight"),
                        w(weights, "fov.head.0.bias"), stride=(2,2), padding=(1,1))
    fov_x = mlx_relu(fov_x)
    fov_x = mlx_conv2d(fov_x, w(weights, "fov.head.2.weight"),
                        w(weights, "fov.head.2.bias"), stride=(2,2), padding=(1,1))
    fov_x = mlx_relu(fov_x)
    fov_x = mlx_conv2d(fov_x, w(weights, "fov.head.4.weight"),
                        w(weights, "fov.head.4.bias"))

    mx.eval(fov_x)

    # ===== Compare =====
    print("\n=== Results ===")
    # MLX depth is NHWC [1, 1536, 1536, 1], PT is NCHW [1, 1, 1536, 1536]
    mlx_depth = np.array(hx)  # [1, 1536, 1536, 1]
    pt_depth_np = pt_depth.numpy()  # [1, 1, 1536, 1536]
    # Transpose MLX to NCHW for comparison
    mlx_depth_nchw = np.transpose(mlx_depth, [0, 3, 1, 2])

    print(f"PT depth range:  [{pt_depth_np.min():.6f}, {pt_depth_np.max():.6f}]")
    print(f"MLX depth range: [{mlx_depth_nchw.min():.6f}, {mlx_depth_nchw.max():.6f}]")

    diff = np.abs(mlx_depth_nchw - pt_depth_np)
    print(f"Max abs diff:  {diff.max():.6e}")
    print(f"Mean abs diff: {diff.mean():.6e}")

    mlx_fov = float(np.array(fov_x).flat[0])
    pt_fov_val = float(pt_fov.numpy().flat[0])
    print(f"PT FOV:  {pt_fov_val:.6f}")
    print(f"MLX FOV: {mlx_fov:.6f}")
    print(f"FOV diff: {abs(mlx_fov - pt_fov_val):.6e}")

    tol = 1e-2
    if diff.max() < tol:
        print(f"\nPASS (tol={tol})")
    else:
        print(f"\nFAIL (tol={tol})")
        # Debug: compare specific elements
        for r, c in [(0,0), (100,100), (768,768)]:
            pv = pt_depth_np[0,0,r,c]
            mv = mlx_depth_nchw[0,0,r,c]
            print(f"  [{r},{c}] pt={pv:.6f} mlx={mv:.6f}")


if __name__ == "__main__":
    main()
