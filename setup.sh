apt update -y
apt install -y \
    python3-pip \
    nano \
    libgl1 \
    npm

npm install -g pm2@latest
pip install -e .