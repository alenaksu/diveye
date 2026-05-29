"""
LU2Net ONNX Export Script
=========================
Exports a LU2Net PyTorch checkpoint → ONNX model.

Supports both checkpoint formats:
  - Raw state_dict  (e.g. LightUNet_170.pth from the original repo)
  - Full checkpoint  (e.g. best.pth produced by train.py, which includes
                      epoch / optimizer / best_psnr alongside state_dict)

Usage
-----
  # Activate venv first
  source .venv/bin/activate

  # Export from original weights (default)
  python export_onnx.py

  # Export from a training run checkpoint
  python export_onnx.py --weights runs/lu2net_20250101_120000/best.pth

  # Export to a custom output path
  python export_onnx.py --weights runs/.../best.pth --output ../public/models/lu2net.onnx
"""

import argparse
import os
import sys
import torch
import onnx
from onnx import checker

sys.path.insert(0, os.path.dirname(__file__))
from LU2Net import LU2Net


DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "LightUNet_170.pth")
DEFAULT_OUTPUT  = os.path.join(os.path.dirname(__file__), "../public/models/lu2net.onnx")


def load_state_dict(path: str, device: str = "cpu") -> dict:
    """Load and unwrap a checkpoint into a plain state_dict."""
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(state)}")

    # Full checkpoint produced by train.py
    if "state_dict" in state:
        sd = state["state_dict"]
        epoch = state.get("epoch", "?")
        psnr  = state.get("best_psnr", None)
        info  = f"epoch={epoch}"
        if psnr is not None:
            info += f", best_psnr={psnr:.2f} dB"
        print(f"  Checkpoint info: {info}")
        return sd

    # Alternative wrapper key used by some training frameworks
    if "model" in state:
        return state["model"]

    # Raw state_dict (original LightUNet_170.pth)
    return state


def export(weights: str, output: str, opset: int = 17) -> None:
    print(f"Loading weights from: {weights}")
    model = LU2Net()
    sd = load_state_dict(weights)
    model.load_state_dict(sd)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}  ({total_params * 4 / 1024 / 1024:.2f} MB fp32)")

    # 256×256 dummy — dynamic axes allow any size at runtime
    # (H and W must be divisible by 4 due to 2× downsampling × 2 layers)
    dummy = torch.zeros(1, 3, 256, 256)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    print("Exporting to ONNX …")
    torch.onnx.export(
        model,
        dummy,
        output,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )

    print("Validating ONNX graph …")
    onnx_model = onnx.load(output)
    checker.check_model(onnx_model)

    size_kb = os.path.getsize(output) / 1024
    print(f"Exported:  {output}  ({size_kb:.1f} KB)")

    # Numerical validation: compare PyTorch vs ONNX Runtime outputs
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        test_input = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            pt_out = model(test_input).numpy()

        ort_out = sess.run(["output"], {"input": test_input.numpy()})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        status = "PASS" if max_diff < 1e-4 else "WARN — diff higher than expected, check results"
        print(f"Max PyTorch/ONNX diff: {max_diff:.2e}  {status}")
    except ImportError:
        print("onnxruntime not installed — skipping numerical validation")

    print("Done.")


def main():
    p = argparse.ArgumentParser(description="Export LU2Net to ONNX")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="Path to .pth checkpoint (raw state_dict or full checkpoint)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Output .onnx path")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version")
    args = p.parse_args()
    export(args.weights, args.output, args.opset)


if __name__ == "__main__":
    main()
