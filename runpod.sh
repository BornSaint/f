# !/bin/bash

start=$(date +%s)

# Detect the number of NVIDIA GPUs and create a device string
gpu_count=$(nvidia-smi -L | wc -l)
if [ $gpu_count -eq 0 ]; then
    echo "No NVIDIA GPUs detected. Exiting."
    exit 1
fi
# Construct the CUDA device string
cuda_devices=""
for ((i=0; i<gpu_count; i++)); do
    if [ $i -gt 0 ]; then
        cuda_devices+=","
    fi
    cuda_devices+="$i"
done

# Install dependencies
apt update
apt install -y screen vim git-lfs
screen

# Install common libraries
pip install -U requests accelerate sentencepiece pytablewriter einops protobuf huggingface_hub==0.21.4
pip install -U transformers
pip install pip3-autoremove
pip-autoremove torch torchvision torchaudio -y
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
pip install unsloth[kaggle-new]
pip uninstall unsloth -y && pip install git+https://github.com/unslothai/unsloth.git
pip install git+https://github.com/unslothai/unsloth-zoo.git
pip-autoremove unsloth unsloth-zoo -y
pip install unsloth
pip install jmespath


# Check if HUGGINGFACE_TOKEN is set and log in to Hugging Face
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "HUGGINGFACE_TOKEN is defined. Logging in..."
    huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
fi

if [ "$DEBUG" == "True" ]; then
    echo "Launch Finetune in debug mode"
fi
cd /f
python ./main.py


if [ "$DEBUG" == "False" ]; then
    runpodctl remove pod $RUNPOD_POD_ID
fi

sleep infinity
