#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) example using INT4 quantization with fixes
for the scale coefficient issues in TensorRT
"""

import argparse
import os
import torch
import numpy as np
from modelopt.torch.quantization import (
    CalibrationDataLoader,
    PostTrainingQuantConfig,
    post_training_quantize,
)
from modelopt.torch._deploy import TensorRTCompileConfig, TensorRTEngineBuilder


def main():
    parser = argparse.ArgumentParser(description="Run PTQ with INT4 quantization (fixed)")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="vit_base_patch16_224.onnx",
        help="Path to the ONNX model"
    )
    parser.add_argument(
        "--calib_data", 
        type=str, 
        default="calib.npy",
        help="Path to calibration data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./quantized_model_int4",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for quantization"
    )
    
    args = parser.parse_args()
    
    # Load calibration data
    if not os.path.exists(args.calib_data):
        print(f"Calibration data not found at {args.calib_data}")
        print("Please run image_prep.py first to generate calibration data")
        return
    
    calib_data = np.load(args.calib_data)
    print(f"Loaded calibration data with shape: {calib_data.shape}")
    
    # Ensure calibration data is properly normalized and positive
    if np.any(calib_data < 0):
        print("Warning: Found negative values in calibration data, applying abs()")
        calib_data = np.abs(calib_data)
    
    # Create calibration data loader
    calib_loader = CalibrationDataLoader(
        calib_data,
        batch_size=args.batch_size,
        input_key="input"
    )
    
    # Configure INT4 quantization with fixes for scale coefficient issues
    quant_config = PostTrainingQuantConfig.from_dict({
        "algorithm": "percentile",  # Use percentile instead of max
        "percentile": 99.9,  # Use 99.9th percentile to avoid extreme values
        "quant_scheme": "symmetric",  # Use symmetric quantization
        "dtype": "int4",  # Keep INT4
        "num_calib_batches": min(50, len(calib_loader)),  # Reduce calibration batches
        "calib_algorithm": "minmax",  # Use minmax calibration
    })
    
    print("Starting INT4 quantization with fixes...")
    
    try:
        # Perform quantization
        quantized_model = post_training_quantize(
            model_path=args.model_path,
            quant_config=quant_config,
            calib_data_loader=calib_loader,
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save quantized model
        quantized_model_path = os.path.join(args.output_dir, "vit_int4_fixed.onnx")
        quantized_model.save(quantized_model_path)
        print(f"Quantized model saved to: {quantized_model_path}")
        
        # Compile with TensorRT (using INT4 with conservative settings)
        print("Compiling with TensorRT...")
        compile_config = TensorRTCompileConfig.from_dict({
            "precision": "int4",  # Keep INT4 precision
            "max_batch_size": args.batch_size,
            "optimization_level": 3,  # Reduce optimization level
            "workspace_size": 1 << 30,  # 1GB workspace
        })
        
        engine_builder = TensorRTEngineBuilder(compile_config)
        engine_path = os.path.join(args.output_dir, "vit_int4_fixed.engine")
        
        engine_builder.build(
            model_path=quantized_model_path,
            engine_path=engine_path
        )
        print(f"TensorRT engine saved to: {engine_path}")
        
    except Exception as e:
        print(f"INT4 quantization failed: {e}")
        print("Falling back to INT8 quantization...")
        
        # Fallback to INT8
        quant_config_int8 = PostTrainingQuantConfig.from_dict({
            "algorithm": "max",
            "quant_scheme": "symmetric",
            "dtype": "int8",
            "num_calib_batches": min(100, len(calib_loader)),
        })
        
        quantized_model = post_training_quantize(
            model_path=args.model_path,
            quant_config=quant_config_int8,
            calib_data_loader=calib_loader,
        )
        
        quantized_model_path = os.path.join(args.output_dir, "vit_int8_fallback.onnx")
        quantized_model.save(quantized_model_path)
        print(f"Fallback INT8 model saved to: {quantized_model_path}")


if __name__ == "__main__":
    main() 