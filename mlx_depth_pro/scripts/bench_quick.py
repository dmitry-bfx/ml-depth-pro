#!/usr/bin/env python3
"""Quick benchmark: PyTorch MPS vs MLX C++ for post-ViT only."""
import time, sys, ctypes, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

print("Loading model...", flush=True)
t0 = time.perf_counter()
from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT
config = DEFAULT_MONODEPTH_CONFIG_DICT
config.checkpoint_uri = "./checkpoints/depth_pro.pt"
model, _ = create_model_and_transforms(config, device=torch.device("cpu"))
model.eval()
print(f"  Model loaded in {time.perf_counter()-t0:.1f}s", flush=True)

# Fake ViT outputs
torch.manual_seed(0)
pe = torch.randn(35, 577, 1024)
h0 = torch.randn(35, 577, 1024)
h1 = torch.randn(35, 577, 1024)
ie = torch.randn(1, 577, 1024)
fv = torch.randn(1, 577, 1024)
img = torch.randn(1, 3, 1536, 1536)
enc = model.encoder

# ===== PyTorch MPS =====
print("\n--- PyTorch MPS ---", flush=True)
dev = torch.device("mps")
enc.to(dev); model.decoder.to(dev); model.head.to(dev); model.fov.to(dev)
pe_d, h0_d, h1_d, ie_d, img_d = pe.to(dev), h0.to(dev), h1.to(dev), ie.to(dev), img.to(dev)

def pt_run():
    with torch.no_grad():
        p = enc.reshape_feature(pe_d, enc.out_size, enc.out_size)
        a = enc.reshape_feature(h0_d, enc.out_size, enc.out_size)
        b = enc.reshape_feature(h1_d, enc.out_size, enc.out_size)
        g = enc.reshape_feature(ie_d, enc.out_size, enc.out_size)
        es = [
            enc.upsample_latent0(enc.merge(a[:25], 1, 3)),
            enc.upsample_latent1(enc.merge(b[:25], 1, 3)),
            enc.upsample0(enc.merge(p[:25], 1, 3)),
            enc.upsample1(enc.merge(p[25:34], 1, 6)),
            enc.fuse_lowres(torch.cat((enc.upsample2(p[34:]), enc.upsample_lowres(g)), dim=1)),
        ]
        f, l = model.decoder(es)
        model.head(f)
        model.fov.forward(img_d, l.detach())
    torch.mps.synchronize()

print("  Warmup...", flush=True)
for _ in range(3):
    pt_run()

times = []
for i in range(5):
    t = time.perf_counter()
    pt_run()
    times.append(time.perf_counter() - t)
    print(f"  Run {i+1}: {times[-1]*1000:.0f} ms", flush=True)
pt_avg = np.mean(times) * 1000
print(f"  Avg: {pt_avg:.0f} ms", flush=True)

# Move back to CPU
enc.to("cpu"); model.decoder.to("cpu"); model.head.to("cpu"); model.fov.to("cpu")
torch.mps.empty_cache()

# ===== MLX C++ =====
print("\n--- MLX C++ ---", flush=True)
lib = ctypes.CDLL("./mlx_depth_pro/build/libdepth_pro_mlx.dylib")
lib.depth_pro_mlx_create.restype = ctypes.c_void_p
lib.depth_pro_mlx_create.argtypes = [ctypes.c_char_p]
lib.depth_pro_mlx_forward.restype = ctypes.c_int
lib.depth_pro_mlx_forward.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)] * 7
lib.depth_pro_mlx_destroy.restype = None
lib.depth_pro_mlx_destroy.argtypes = [ctypes.c_void_p]

m = lib.depth_pro_mlx_create(b"./mlx_depth_pro/weights/mlx_weights.safetensors")
assert m

pen = np.ascontiguousarray(pe.numpy(), dtype=np.float32)
h0n = np.ascontiguousarray(h0.numpy(), dtype=np.float32)
h1n = np.ascontiguousarray(h1.numpy(), dtype=np.float32)
ien = np.ascontiguousarray(ie.numpy(), dtype=np.float32)
fvn = np.ascontiguousarray(fv.numpy(), dtype=np.float32)
db = np.zeros((1, 1, 1536, 1536), dtype=np.float32)
fb = np.zeros(1, dtype=np.float32)
cp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def mlx_run():
    lib.depth_pro_mlx_forward(m, cp(pen), cp(h0n), cp(h1n), cp(ien), cp(fvn), cp(db), cp(fb))

print("  Warmup...", flush=True)
for _ in range(3):
    mlx_run()

times2 = []
for i in range(5):
    t = time.perf_counter()
    mlx_run()
    times2.append(time.perf_counter() - t)
    print(f"  Run {i+1}: {times2[-1]*1000:.0f} ms", flush=True)
mlx_avg = np.mean(times2) * 1000
print(f"  Avg: {mlx_avg:.0f} ms", flush=True)

print(f"\n=== Result: MLX is {pt_avg/mlx_avg:.2f}x vs PyTorch MPS ===")
lib.depth_pro_mlx_destroy(m)
