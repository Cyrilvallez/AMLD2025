sudo apt update

# Install gcc
sudo apt install -y gcc

sudo apt install -y linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install nvidia drivers
sudo apt install -y nvidia-driver-550

# Install cuda toolkit
sudo apt install -y cuda-toolkit-12-8

# Add cuda to path
echo -e '\nexport PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> .bashrc

# Remove keyring
rm cuda-keyring_1.1-1_all.deb

sudo reboot