#!/bin/bash

set -a
source miner.env
set +a

MINER_PROCESS_NAME="climate_miner"


if pm2 list | grep -q "$MINER_PROCESS_NAME"; then
  echo "Process '$MINER_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $MINER_PROCESS_NAME
fi

pm2 start neurons/miner.py --name $MINER_PROCESS_NAME -- \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $AXON_PORT \
  --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT \
  --logging.info

#python3 neurons/miner.py --subtensor.chain_endpoint ws://127.0.0.1:9944 --netuid 1 --wallet.name validator --wallet.hotkey miner --axon.port 8080 --logging.info