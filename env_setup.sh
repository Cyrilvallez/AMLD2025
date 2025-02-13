# Download and install Miniforge (an equivalent of Miniconda)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O ~/miniforge.sh
bash ~/miniforge.sh -b -p ~/miniforge3

# Activate base env and run init for the future
source ~/miniforge3/etc/profile.d/conda.sh
conda activate
conda init

# Delete installer
rm ~/miniforge.sh

# Create env
conda create -n amld python=3.12 -y

# Activate env
conda activate amld

# Install packages in env
pip install torch
pip install transformers accelerate huggingface-hub gradio
pip install sentencepiece tiktoken protobuf