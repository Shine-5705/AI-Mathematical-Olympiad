"""
Convert HuggingFace models to MLX format for Mac training.

Usage:
    python convert_model.py
    python convert_model.py --model AI-MO/NuminaMath-7B-TIR --quantize
    python convert_model.py --model Qwen/Qwen2.5-Math-7B-Instruct --bits 4
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to MLX format")
    parser.add_argument("--model", default="AI-MO/NuminaMath-7B-TIR")
    parser.add_argument("--output", default=None, help="Output dir (default: models/mlx/<model-name>)")
    parser.add_argument("--quantize", action="store_true", default=True, help="Quantize to 4-bit")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    output_dir = args.output or f"models/mlx/{model_name}-{args.bits}bit"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", args.model,
        "--mlx-path", output_dir,
    ]

    if args.quantize:
        cmd += ["-q", "--q-bits", str(args.bits)]

    print(f"Converting: {args.model}")
    print(f"Output:     {output_dir}")
    print(f"Quantize:   {'yes' if args.quantize else 'no'} ({args.bits}-bit)")
    print()
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\nDone. Use this model with:")
        print(f"  python train.py --mode sft --model {output_dir}")
    else:
        print(f"\nConversion failed (exit code {result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    main()
