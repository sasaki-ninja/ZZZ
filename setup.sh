#!/bin/bash

# check if sudo exists as it doesn't on RunPod
if command -v sudo 2>&1 >/dev/null
then
    PREFIX="sudo"
else
    PREFIX=""
fi

$PREFIX apt update -y
$PREFIX apt install -y \
    python3-pip \
    nano \
    libgl1 \
    npm

$PREFIX npm install -g pm2@latest

# install repository itself
pip install -e . --use-pep517