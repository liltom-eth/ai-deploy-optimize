# ai-deploy-optimize

## Env
### Docker

Build the Docker image (will be tagged ai_deploy_optimize:latest)

```bash
chmod +x docker/build.sh
./docker/build.sh
```

Run the docker image
```bash
docker run --user 0:0 -it --gpus all --shm-size=2g -v /home/tom/datasets:/workspace/datasets  -v /home/tom/projects/TensorRT-Model-Optimizer:/workspace/TensorRT-Model-Optimizer ai_deploy_optimize:25.04 bash
```

## PTQ
Prepare the example model
Most of the examples in this doc use vit_base_patch16_224.onnx as the input model. The model can be downloaded with the following script:

```bash
python download_vit_onnx.py \
    --onnx_save_path=vit_base_patch16_224.onnx \
    --fp16 `# <Optional, if the desired output ONNX precision is FP16>`
```

First, prepare some calibration data. TensorRT recommends calibration data size to be at least 500 for CNN and ViT models. The following command picks up 500 images from the tiny-imagenet dataset and converts them to a numpy-format calibration array. Reduce the calibration data size for resource constrained environments.

```bash
python image_prep.py \
    --calibration_data_size=500 \
    --output_path=calib.npy \
    --fp16 `# <Optional, if the input ONNX is in FP16 precision>`
```
For Int4 quantization, it is recommended to set --calibration_data_size=64.