# Validator Guide

## Table of Contents

1. [Installation 🔧](#installation)
   - [Registration ✍️](#registration)
2. [Validating ✅](#validating)
   - [ECMWF 🌎](#ecmwf)
3. [Requirements 💻](#requirements)

## Before you proceed ⚠️

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

## Installation
> [!TIP]
> If you are using RunPod, you can use our [dedicated template](https://runpod.io/console/deploy?template=cyui16nkkd&ref=97t9kcqz) (or use the [Miner template](https://runpod.io/console/deploy?template=x2lktx2xex&ref=97t9kcqz) for GPU support) which comes pre-installed with all required dependencies! Even without RunPod the [Docker image](https://hub.docker.com/repository/docker/ericorpheus/zeus/) behind this template might still work for your usecase.

If you are using the Docker image, you still need to clone the GitHub repository. However, all required libraries should be pre-installed within the Docker environment. Therefore, you can skip the Conda virtual environment setup.

Download the repository and navigate to the folder.
```bash
git clone https://github.com/Orpheus-AI/Zeus.git && cd Zeus
```

If you are **not** using the Docker image, we recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install), and create a virtual environment with this command:

```bash
conda create -y -n zeus python=3.11
```

To activate your virtual environment, run `conda activate zeus`. To deactivate, `conda deactivate`.

Install the remaining necessary requirements with the following chained command.

```bash
conda activate zeus
chmod +x setup.sh
./setup.sh
```

## Registration

To validate on our subnet, you must have a registered hotkey.

#### Mainnet

```bash
btcli s register --netuid [net_uid] --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 301 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```


## Validating
Before launching your validator, make sure to create a file called `validator.env`. This file will not be tracked by git. 
You can use the sample below as a starting point, but make sure to replace **wallet_name**, **wallet_hotkey**, **axon_port**, **wandb_api_key** and **cds_api_key**.

```bash
NETUID=301                                      # Network User ID options: ?,301
SUBTENSOR_NETWORK=test                          # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443/
                                                # Endpoints:
                                                # - wss://entrypoint-finney.opentensor.ai:443
                                                # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Validator Port Setting:
AXON_PORT=8092
PROXY_PORT=10913

# API Keys:
WANDB_API_KEY=your_wandb_api_key_here
CDS_API_KEY=your_cds_api_key_here
```
If you don't have a W&B API key, please reach out to Ørpheus A.I. via Discord. Without W&B, miners will not be able to see their live scores, 
so we highly recommend enabling this.


### ECMWF
> [!IMPORTANT]
> In order to send miners challenges involving the latest ERA5 data, you need to provide a Copernicus CDS API key. The steps below explain how to obtain this key. If you encounter any difficulty in the process, please let us know and we will create an account for you.

1. Go the the official [CDS website](https://cds.climate.copernicus.eu/how-to-api).
2. Click on the "Login - Register" button in the top right of the page.
3. Click the "I understand" button on the screen that pops up to be redirected to the next page.
4. Unless you already have an account, click the blue "Register" button in the gray box below the login page.
5. Fill in your details and complete the Captcha. Keep in mind that you need to be able to access the email address used. Then click the blue register button.
6. Go to your email and click the link in the email from `servicedesk@ecmwf.int`. You should be taken to a page to enter more information. If not, go the link from step 1 and try to login instead of registering. 
7. Fill in the extra details (they are not checked at all and don't have to be accurate) and accept the statements. Click the "activate your profile" button.
8. You should be redirected back to the [CDS website](https://cds.climate.copernicus.eu/how-to-api). Scroll down to the section labeled '1. Setup the CDS API personal access token.' You will find a code block containing your API key. **Crucially, copy only the value of the 'key' portion of this code block into your `validator.env` file.**
    For example, the code block will resemble the following:

    ```
    url: https://cds.climate.copernicus.eu/api
    key: YOUR_API_KEY_THAT_SHOULD_BE_COPIED
    ```

    **Only copy the string following 'key:' (i.e., `YOUR_API_KEY_THAT_SHOULD_BE_COPIED`) into your environment file.**
9. Please ensure you accept the [terms for downloading ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences), as this is the dataset used for validator queries.

Now you're ready to run your validator!

```bash
conda activate zeus
pm2 start run_neuron.py -- --validator 
```

- Auto updates are enabled by default. To disable, run with `--no-auto-updates`.
- Self-healing restarts are disabled by default (every 3 hours). To enable, run with `--self-heal`.

## Requirements
We strive to make validation as simple as possible on our subnet, aiming to minimise storage and hardware requirements for our validators.
Only a couple days of environmental data need to be stored at a time, which will never exceed 1GB. Miner predictions are also temporarily stored in an SQLite database for challenges where the ground-truth is not yet known, which should also not exceed 1GB. As long as you have enough storage to install our standard Python dependencies (i.e. PyTorch), you can likely run our entire codebase!  

Data processing is done locally, but since this has been highly optimised, you will also **not need any GPU** or CUDA support. You will only need a decent CPU machine, where we recommend having at least 8GB of RAM. Since data is loaded over the internet, it is useful to have at least a moderately decent (>3MB/s) internet connection.

You are required to provide an API key for the Climate Data Store in order to retrieve data to send to miners, the validator will shut down if this authentication fails. The API-key can be retrieved from the [official CDS website](https://cds.climate.copernicus.eu/how-to-api), after you have created an account at this link as well. Creating an account is **completely free of charge** and only necessary at first launch. Once you have obtained an API key, please enter it in the [validator.env](../validator.env) file. 

We would kindly ask you to link you validator to Weights and Biases, since this helps both miners and outside parties to obtain visualisation of the current state of the subnet. This can be done by specifying your API key in the ``validator.env` file.

> [!TIP]
> Should you need any assistance with setting up the validator, W&B or anything else, please don't hesitate to reach out to the team at Ørpheus A.I. via Discord!
