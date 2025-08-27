# Use the TensorRT base image with version tensorrt:10.9.0.34
FROM nvcr.io/nvidia/tensorrt:25.04-py3

# Update PATH and LD_LIBRARY_PATH variables for the TensorRT binaries
# ENV LD_LIBRARY_PATH="/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:${LD_LIBRARY_PATH}" \
#     PATH="/usr/local/tensorrt/targets/x86_64-linux-gnu/bin:${PATH}"

# Export the path to 'libcudnn.so.X' needed by 'libonnxruntime_providers_tensorrt.so'
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install ModelOpt and precompile the extensions
RUN pip install -U "nvidia-modelopt[all,dev-test]"
RUN python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

# Install timm
RUN pip install timm

# Create workspace directory and allow users to run without root
RUN mkdir -p /workspace && chmod -R 777 /workspace

# Set working directory
WORKDIR /workspace
