from fastapi import FastAPI, HTTPException, Depends, Request
from concurrent.futures import ThreadPoolExecutor
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
import bittensor as bt
import traceback
import pandas as pd
import uvicorn
import base64
import torch
import asyncio
import traceback
import httpx
import base64

from zeus.utils.uids import get_random_uids
from zeus.validator.reward import help_format_miner_output, compute_penalty
from zeus.protocol import TimePredictionSynapse
from zeus.utils.time import get_timestamp
from zeus.utils.coordinates import get_grid

class ValidatorProxy:
    def __init__(
        self,
        validator,
    ):
        self.validator = validator
        #self.get_credentials()
        self.dendrite = bt.dendrite(wallet=validator.wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/proxy",
            self.proxy,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_api_route(
            "/healthcheck",
            self.healthcheck,
            methods=["GET"],
            dependencies=[Depends(self.get_self)],
        )

        self.loop = asyncio.get_event_loop()
        if self.validator.config.proxy.port:
            self.start_server()

    def get_credentials(self):
        with httpx.Client(timeout=httpx.Timeout(30)) as client:
            response = client.post(
                f"{self.validator.config.proxy.proxy_client_url}/get-credentials",
                json={
                    "postfix": (
                        f":{self.validator.config.proxy.port}/validator_proxy"
                        if self.validator.config.proxy.port
                        else ""
                    ),
                    "uid": self.validator.uid,
                },
            )
        response.raise_for_status()
        response = response.json()
        message = response["message"]
        signature = response["signature"]
        signature = base64.b64decode(signature)

        def verify_credentials(public_key_bytes):
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            try:
                public_key.verify(signature, message.encode("utf-8"))
            except InvalidSignature:
                raise Exception("Invalid signature")

        self.verify_credentials = verify_credentials

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run, self.app, host="0.0.0.0", port=self.validator.config.proxy.port
        )

    def authenticate_token(self, public_key_bytes):
        public_key_bytes = base64.b64decode(public_key_bytes)
        try:
            self.verify_credentials(public_key_bytes)
            bt.logging.info("Successfully authenticated token")
            return public_key_bytes
        except Exception as e:
            bt.logging.error(f"Exception occured in authenticating token: {e}")
            bt.logging.error(traceback.print_exc())
            raise HTTPException(
                status_code=401, detail="Error getting authentication token"
            )

    async def healthcheck(self, request: Request):
        authorization: str = request.headers.get("authorization")

        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        self.authenticate_token(authorization)
        return {'status': 'healthy'}

    async def proxy(self, request: Request):
        authorization: str = request.headers.get("authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        #self.authenticate_token(authorization) # TODO

        bt.logging.info("[PROXY] Received an organic request!")
        # Finding some miners
        metagraph = self.validator.metagraph
        miner_uids = self.validator.last_responding_miner_uids
        if len(miner_uids) == 0:
            bt.logging.warning("[PROXY] No recent miner uids found, sampling random uids")
            miner_uids = get_random_uids(self.validator, k=self.validator.config.neuron.sample_size)

        # catch errors to prevent log spam if API is missused
        try:
            payload = await request.json()
           
            grid = get_grid(payload['lat_start'], payload['lat_end'], payload['lon_start'], payload['lon_end'])
            predict_hours = payload["predict_hours"]
            synapse = TimePredictionSynapse(
                locations=grid.tolist(), 
                start_time=payload['start_timestamp'], 
                end_time=payload['end_timestamp'],
                requested_hours=predict_hours
            )

        except Exception as e:
            bt.logging.info(f"[PROXY] Organic request was invalid.")
            error_msg = traceback.format_exc()
            raise HTTPException(status_code=400, detail=f"Invalid request, parsing failed with error:\n {error_msg}")

        # getting some responses from miners
        bt.logging.info(f"[PROXY] Querying {len(miner_uids)} miners...")
        predictions = await self.dendrite(
            axons=[metagraph.axons[uid] for uid in miner_uids],
    	    synapse=synapse,
            deserialize=True,
            timeout=10
        )

        # validating predictions and returning them
        dummy_output = torch.zeros(predict_hours, grid.shape[0], grid.shape[1])
        for prediction, uid in zip(predictions, miner_uids):
            prediction = help_format_miner_output(dummy_output, prediction)
            penalty = compute_penalty(dummy_output, prediction)
            if penalty > 0:
                continue

            bt.logging.info(f"[PROXY] Obtained a valid prediction from miner {uid}")

            # miners were sorted in order of performance on last prediction, so return first valid one.
            return {
                'prediction': prediction.tolist()
            }
        
        return HTTPException(status_code=500, detail="No valid response received")

    async def get_self(self):
        return self