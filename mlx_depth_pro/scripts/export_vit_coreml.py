#!/usr/bin/env python3
"""Export DINOv2 ViT backbones to CoreML for Depth Pro.

Exports 3 models:
  - patch_encoder: [35, 3, 384, 384] -> output [35, 577, 1024] + hook0 [35, 577, 1024] + hook1 [35, 577, 1024]
  - image_encoder: [1, 3, 384, 384] -> [1, 577, 1024]
  - fov_encoder: [1, 3, 384, 384] -> [1, 577, 1024]

The patch_encoder needs to output intermediate block features (hooks).
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth_pro.network.vit_factory import create_vit, VIT_CONFIG_DICT


class PatchEncoderWithHooks(nn.Module):
    """Wraps the ViT patch encoder to explicitly return intermediate features."""

    def __init__(self, vit_model, hook_block_ids):
        super().__init__()
        self.model = vit_model
        self.hook_ids = hook_block_ids
        self._hooks = {}

        for idx in self.hook_ids:
            self.model.blocks[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx):
        def fn(mod, inp, out):
            self._hooks[idx] = out
        return fn

    def forward(self, x):
        self._hooks.clear()
        out = self.model(x)
        hook0 = self._hooks[self.hook_ids[0]]
        hook1 = self._hooks[self.hook_ids[1]]
        return out, hook0, hook1


def export_model(model, example_input, name, output_dir):
    """Trace and export a model to CoreML."""
    model.eval()
    traced = torch.jit.trace(model, example_input)

    if isinstance(example_input, tuple):
        inputs = [ct.TensorType(name=f"input_{i}", shape=inp.shape) for i, inp in enumerate(example_input)]
    else:
        inputs = [ct.TensorType(name="input", shape=example_input.shape)]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    out_path = Path(output_dir) / f"{name}.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/depth_pro.pt")
    parser.add_argument("--output-dir", default="./mlx_depth_pro/weights")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = VIT_CONFIG_DICT["dinov2l16_384"]
    hook_ids = config.encoder_feature_layer_ids[:2]  # [5, 11]

    print("Loading checkpoint...")
    full_sd = torch.load(args.checkpoint, map_location="cpu")

    def load_vit(prefix):
        model = create_vit("dinov2l16_384", use_pretrained=False)
        state = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    # --- Patch encoder (with hooks for intermediate features) ---
    print("\nExporting patch_encoder (with hooks)...")
    patch_vit = load_vit("encoder.patch_encoder.")
    patch_model = PatchEncoderWithHooks(patch_vit, hook_ids)
    patch_model.eval()

    # CoreML doesn't support batch=35 well with hooks via tracing.
    # Export with batch=1 and call 35 times, or export with batch=35.
    # Let's try batch=35 first.
    try:
        example = torch.randn(35, 3, 384, 384)
        export_model(patch_model, example, "patch_encoder", args.output_dir)
    except Exception as e:
        print(f"  Batch=35 export failed: {e}")
        print("  Trying batch=1 (will need to call 35 times)...")
        example = torch.randn(1, 3, 384, 384)
        export_model(patch_model, example, "patch_encoder_b1", args.output_dir)

    # --- Image encoder ---
    print("\nExporting image_encoder...")
    image_vit = load_vit("encoder.image_encoder.")
    example = torch.randn(1, 3, 384, 384)
    export_model(image_vit, example, "image_encoder", args.output_dir)

    # --- FOV encoder ---
    print("\nExporting fov_encoder...")
    fov_vit = load_vit("fov.encoder.0.")
    example = torch.randn(1, 3, 384, 384)
    export_model(fov_vit, example, "fov_encoder", args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
