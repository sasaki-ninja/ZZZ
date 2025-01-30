# Miner Guide

## Table of Contents

1. [Installation 🔧](#installation)
2. [Registration ✍️](#registration)
3. [Setup ⚙️](#setup)
3. [Mining ⛏️](#mining)

## Before you proceed ⚠️

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml). A GPU is required for training (unless you want to wait weeks for training to complete), but is not required for inference while running a miner.

## Installation

Download the repository and navigate to the folder.
```bash
git clone https://github.com/Orpheus-AI/ClimateAI.git && cd ClimateAI
```

We recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install). Note that after you run the last commands in the miniconda setup process, you'll be prompted to start a new shell session to complete the initialization. 

With miniconda installed, you can create a virtual environment with this command:

```bash
conda create -y -n climate python=3.11
```

To activate your virtual environment, run `conda activate climate`. To deactivate, `conda deactivate`.

Install the remaining necessary requirements with the following chained command. This may take a few minutes to complete.

```bash
conda activate climate
chmod +x setup.sh 
./setup.sh
```


## Registration

To mine on our subnet, you must have a registered hotkey.

*Note: For testnet tao, you can make requests in the [Bittensor Discord's "Requests for Testnet Tao" channel](https://discord.com/channels/799672011265015819/1190048018184011867)*

To reduce the risk of deregistration due to technical issues or a poor performing model, we recommend the following:
1. Test your miner on testnet before you start mining on mainnet.
2. Before registering your hotkey on mainnet, make sure your port is open by running `curl your_ip:your_port`
3. If you've trained a custom model, test it's performance by deploying to testnet. Testnet performance is logged to a dedicated [Weights and Biases](https://wandb.ai/orpheus-ai/climate-subnet).


#### Mainnet

```bash
btcli s register --netuid [net_uid] --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 301 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

## Setup
Before launching your miner, make sure to create a file called `miner.env`. This file will not be tracked by git. 
You can use the sample below as a starting point, but make sure to replace **wallet_name**, **wallet_hotkey**, and **axon_port**.


```bash
# Subtensor Network Configuration:
NETUID=301                                     # Network User ID options: ?,301
SUBTENSOR_NETWORK=                             # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                               # Endpoints:
                                               # - wss://entrypoint-finney.opentensor.ai:443
                                               # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Miner Settings:
AXON_PORT=8091
BLACKLIST_FORCE_VALIDATOR_PERMIT=True          # Default setting to force validator permit for blacklisting
```

## Mining
Now you're ready to run your miner!

```bash
conda activate climate
./start_miner.sh
```

The base miner already provides a very competitive level of predictions. It invokes [OpenMeteo's API](https://open-meteo.com/), which uses current state-of-the-art models to do forecasting. While improving upon this prediction might initially be challenging, just a small improvement will mean you can potentially obtain far higher rewards. Also note that you might run out of free-tier credits for this API eventually, depending on how often your miner is queried.

### Input and desired output data
The datasource for this subnet consists of ERA5 reanalysis data from the Climate Data Store (CDS) of the European Union's Earth observation programme (Copernicus). This comprises the largest global environmental dataset to date, containing hourly measurements across a multitude of variables. 

In short, the miners will be send a 2D grid of latitude and longitude locations and a time range **and** are asked to perform hourly forecasts at all those locations. Initially we will focus on predicting temperature two meters above the earth's surface, in the form of the `2m_temperature` variable. Both the time interval and the geographical location will be randomly chosen, and both can be of **variable** size (within some [constraints](../climate/validator/constants.py)). You will be asked to predict hourly temperature measurements for these exact locations, for different numbers of hours.

The locations will be send to you in the form of a **3D tensor** (converted to stacked lists of floats), with the following axes in order: latitude, longitude, variables. Variables consists of 2 values, with the first being the latitudinal coordinate and the second the longitudinal coordinate. Latitude is a float between -90 and 90, whereas longitude spans -180 to 180. Note that you will not be send a global earth representation, but rather a specific slice of this maximal range. 

Secondly, you will be send the start and end timestamp of the time-range you need to predict. These are formatted as float-timestamps, according to timezone GMT+0 (timezone used by Copernicus). These values will always be exactly rounded at the hour. You are also separately send the number of hours you need to predict as an integer (you could calculate this yourself based on the timestamps). 

You should predict the temperature **in degrees Kelvin** value for each hour in the time-range, for all input locations. Note that you do not need to send back the location data itself. So the ideal return format is therefore a **3D tensor** (as a stacked list of floats) with the following axes: requested_output_hours, latitude, longitude. The value in each slot of this tensor corresponds to the temperature you are predicting at that location. The [default miner code](../neurons/miner.py) illustrates exactly how to handle the input-output datastream of this subnet. 

You will be scored based on the Root Mean Squared Error (RMSE) between your temperature predictions and the actual temperature at those locations for the timepoints you were requested. This actualy temperature is not yet known at the time you are send the challenge, meaning you will be scored at a future time. Your goal is to minimise this RMSE, which will increase your final score and subnet incentive. Depending on the difficulty of the challenge you are send (i.e. the variability of the region you need to predict), your RMSE is weighted differently to ensure fairer playing field. 

> [!IMPORTANT]
> Both the number of hours and the size of the grid your miner needs to process will vary across requests. Your miner will therefore need to be able to handle variable-sized inputs of arbitrary size. Whether your miner will internally maintain the grid representation, or flatten time, latitude and longitude into a singular collection is up to you. If you do decide to flatten this grid, make sure to reformat your final output to be of the correct grid-like structure again.




