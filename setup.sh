sudo apt update -y
sudo apt install -y \
    python3-pip \
    nano \
    libgl1 \
    npm

sudo npm install -g pm2@latest
pip install -e .