#!/usr/bin/env bash
# FoundationPose on the GPU workstation (Linux + NVIDIA GPU + Docker with the NVIDIA container toolkit).
# Follows the official README (github.com/NVlabs/FoundationPose, "Env setup option 1: docker").
# Run the steps one by one the first time; paths are examples.
set -e

WORK=${WORK:-$HOME/leonardo}
mkdir -p "$WORK" && cd "$WORK"

# 1. code: the official repository next to this project
[ -d FoundationPose ] || git clone https://github.com/NVlabs/FoundationPose.git
[ -d oscilloscope-detection ] || echo "copy or clone this project to $WORK/oscilloscope-detection"

# 2. network weights (README: download from its Google Drive link and put them in FoundationPose/weights/):
#    weights/2023-10-28-18-33-37  (refiner)   and   weights/2024-01-11-20-02-45  (scorer)
ls FoundationPose/weights || echo "!! download the weights into $WORK/FoundationPose/weights first"

# 3. image (official). For RTX 40xx the README points to: shingarey/foundationpose_custom_cuda121:latest
docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose

# 4. start a container with both repositories mounted
docker run --gpus all -it --rm --network host --ipc host \
  -v "$WORK/FoundationPose:/FoundationPose" -v "$WORK/oscilloscope-detection:/app" \
  -e FOUNDATIONPOSE_DIR=/FoundationPose -e POSE_BACKEND=foundationpose \
  foundationpose bash -lc '
    cd /FoundationPose && bash build_all.sh                       # first time: builds the CUDA extensions
    pip install fastapi uvicorn python-multipart loguru tabulate thop pycocotools pyyaml
    # 5a. sanity check: the official demo on our synthetic TDS 1002 scene (ground truth known)
    python run_demo.py --mesh_file /app/datasets/pose_test/tek_tds1002/mesh/tek_tds1002.obj \
                       --test_scene_dir /app/datasets/pose_test/tek_tds1002 --debug 1
    # 5b. our pose stage with FoundationPose, scored against the ground truth
    cd /app && python tools/pose_eval.py datasets/pose_test/tek_tds1002 --backend foundationpose --overlay
    # 6. the server for the Quest / web page (LAN only)
    cd /app && python -m uvicorn server.app:app --host 0.0.0.0 --port 8011
  '
