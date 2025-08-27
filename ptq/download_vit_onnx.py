import argparse
import os
import subprocess

import timm
import torch

from modelopt.torch._deploy.utils import get_onnx_bytes


def export_to_onnx(
    model, input_shape, onnx_save_path, device, weights_dtype="float32", use_autocast=False
):
    """Export the torch model to ONNX format."""
    # Create input tensor with same precision as model's first parameter
    input_dtype = model.parameters().__next__().dtype
    input_tensor = torch.randn(input_shape, dtype=input_dtype).to(device)

    onnx_model_bytes = get_onnx_bytes(
        model=model,
        dummy_input=(input_tensor,),
        weights_dtype=weights_dtype,
        use_autocast=use_autocast,
    )

    # Write ONNX model to disk
    with open(onnx_save_path, "wb") as f:
        f.write(onnx_model_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and export example models to ONNX.")
    parser.add_argument(
        "--onnx_save_path", type=str, required=False, help="Path to save the final ONNX model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the exported ViT model.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to export the ONNX model in FP16.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000).to(
        device
    )
    data_config = timm.data.resolve_model_data_config(model)
    input_shape = (args.batch_size,) + data_config["input_size"]

    vit_save_path = args.onnx_save_path or "vit_base_patch16_224.onnx"
    weights_dtype = "float16" if args.fp16 else "float32"
    export_to_onnx(
        model,
        input_shape,
        vit_save_path,
        device,
        weights_dtype=weights_dtype,
    )
    print(f"ViT model exported to {vit_save_path}")
