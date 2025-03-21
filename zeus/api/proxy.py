import os
import base64
import asyncio
import traceback

import bittensor as bt
import pandas as pd
import pytz
import torch
import uvicorn

from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from fastapi import FastAPI, HTTPException, Depends, Request
from timezonefinder import TimezoneFinder

from zeus.utils.uids import get_random_uids
from zeus.validator.reward import help_format_miner_output, compute_penalty
from zeus.protocol import TimePredictionSynapse
from zeus.utils.time import get_timestamp
from zeus.utils.coordinates import get_grid, get_closest_grid_points


class ValidatorProxy:
    def __init__(
        self,
        validator,
    ):
        load_dotenv(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../validator.env")
        )
        self.proxy_api_key = os.getenv("PROXY_API_KEY")
        self.tf = TimezoneFinder()
        self.validator = validator
        self.dendrite = bt.dendrite(wallet=validator.wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/predictGridTemperature",
            self.predict_grid_temperature,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_api_route(
            "/predictPointTemperature",
            self.predict_point_temperature,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )

        self.loop = asyncio.get_event_loop()
        if self.validator.config.proxy.port:
            self.start_server()

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run, self.app, host="0.0.0.0", port=self.validator.config.proxy.port
        )

    def authorize_token(self, headers):
        authorization: str = headers.get("authorization", None)
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        if authorization != self.proxy_api_key:
            raise HTTPException(status_code=401, detail="Invalid authorization token")

    async def predict_grid_temperature(self, request: Request):
        self.authorize_token(request.headers)
        bt.logging.info("[PROXY] Received an organic request!")
        # Finding some miners
        metagraph = self.validator.metagraph
        miner_uids = self.validator.last_responding_miner_uids
        if len(miner_uids) == 0:
            bt.logging.warning(
                "[PROXY] No recent miner uids found, sampling random uids"
            )
            miner_uids = get_random_uids(
                self.validator, k=self.validator.config.neuron.sample_size
            )

        # catch errors to prevent log spam if API is missused
        try:
            payload = await request.json()
            lat_start, lat_end = payload.get("lat_start", None), payload.get(
                "lat_end", None
            )
            lon_start, lon_end = payload.get("lon_start", None), payload.get(
                "lon_end", None
            )
            if (
                lat_start is None
                or lat_end is None
                or lon_start is None
                or lon_end is None
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid request, lat_start, lat_end, lon_start and lon_end are required",
                )

            grid = get_grid(lat_start, lat_end, lon_start, lon_end)

            # get middle of the grid
            lat = (lat_start + lat_end) / 2
            lon = (lon_start + lon_end) / 2

            timezone_name = self.tf.timezone_at(lng=lon, lat=lat)
            timezone = pytz.timezone(timezone_name)
            start_timestamp, end_timestamp, predict_hours, timestamps = await self._handle_time_inputs(
                payload, timezone
            )

            synapse = TimePredictionSynapse(
                locations=grid.tolist(),
                start_time=start_timestamp,
                end_time=end_timestamp,
                requested_hours=predict_hours,
            )

        except Exception as e:
            bt.logging.info(f"[PROXY] Organic request was invalid.")
            error_msg = traceback.format_exc()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request, parsing failed with error:\n {error_msg}",
            )

        # getting some responses from miners
        predictions = await self.dendrite(
            axons=[metagraph.axons[uid] for uid in miner_uids],
            synapse=synapse,
            deserialize=True,
            timeout=10,
        )

        bt.logging.info(f"[PROXY] Received {len(predictions)} responses from miners")
        for prediction in predictions:
            bt.logging.info(f"[PROXY] Prediction shape: {prediction.shape}")

        # validating predictions and returning them
        dummy_output = torch.zeros(predict_hours, grid.shape[0], grid.shape[1])
        for prediction, uid in zip(predictions, miner_uids):
            prediction = help_format_miner_output(dummy_output, prediction)
            penalty = compute_penalty(dummy_output, prediction)
            if penalty > 0:
                # if there is a penalty, we skip this prediction
                continue

            bt.logging.info(f"[PROXY] Obtained a valid prediction from miner {uid}")
            return {
                "prediction": prediction.tolist(),
                "grid": grid.tolist(),
                "timestamps": timestamps,
            }

        return HTTPException(status_code=500, detail="No valid response received")

    async def predict_point_temperature(self, request: Request):
        self.authorize_token(request.headers)
        bt.logging.info("[PROXY] Received an organic request!")
        # Finding some miners
        metagraph = self.validator.metagraph
        miner_uids = self.validator.last_responding_miner_uids
        if len(miner_uids) == 0:
            bt.logging.warning(
                "[PROXY] No recent miner uids found, sampling random uids"
            )
            miner_uids = get_random_uids(
                self.validator, k=self.validator.config.neuron.sample_size
            )

        # catch errors to prevent log spam if API is missused
        try:
            payload = await request.json()
            lat, lon = payload.get("lat", None), payload.get("lon", None)
            if lat is None or lon is None:
                raise HTTPException(
                    status_code=400, detail="Invalid request, lat and lon are required"
                )

            grid = get_closest_grid_points(lat, lon)

            timezone_name = self.tf.timezone_at(lng=lon, lat=lat)
            timezone = pytz.timezone(timezone_name)
            start_timestamp, end_timestamp, predict_hours, timestamps = await self._handle_time_inputs(
                payload, timezone
            )

            synapse = TimePredictionSynapse(
                locations=grid.tolist(),
                start_time=start_timestamp,
                end_time=end_timestamp,
                requested_hours=predict_hours,
            )

        except Exception as e:
            bt.logging.info(f"[PROXY] Organic request was invalid.")
            bt.logging.info(f"Error: {e}")
            error_msg = traceback.format_exc()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request, parsing failed with error:\n {error_msg}",
            )

        # getting some responses from miners
        bt.logging.info(f"[PROXY] Querying {len(miner_uids)} miners...")
        predictions = await self.dendrite(
            axons=[metagraph.axons[uid] for uid in miner_uids],
            synapse=synapse,
            deserialize=True,
            timeout=10,
        )

        bt.logging.info(f"[PROXY] Received {len(predictions)} responses from miners")
        for prediction in predictions:
            bt.logging.info(f"[PROXY] Prediction shape: {prediction.shape}")

        dummy_output = torch.zeros(predict_hours, grid.shape[0], grid.shape[1])
        for prediction, uid in zip(predictions, miner_uids):
            prediction = help_format_miner_output(dummy_output, prediction)
            penalty = compute_penalty(dummy_output, prediction)
            if penalty > 0:
                continue

            prediction_shape = prediction.shape
            return await self._interpolate_prediction(
                prediction, prediction_shape, grid, lat, lon, timestamps
            )

        return HTTPException(status_code=500, detail="No valid response received")

    async def _handle_time_inputs(self, payload, timezone):
        start_timestamp_input = payload.get("start_timestamp", None)
        end_timestamp_input = payload.get("end_timestamp", None)
        predict_hours_input = payload.get("predict_hours", None)

        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        if (
            start_timestamp_input is None
            and end_timestamp_input is None
            and predict_hours_input is not None
        ):
            predict_hours = int(predict_hours_input)
            start_timestamp = current_hour
            end_timestamp = start_timestamp + timedelta(hours=predict_hours)

        elif (
            start_timestamp_input is not None
            and end_timestamp_input is None
            and predict_hours_input is None
        ):
            start_timestamp = datetime.fromtimestamp(start_timestamp_input)
            end_timestamp = start_timestamp + timedelta(hours=24)
            predict_hours = 24

        elif (
            start_timestamp_input is not None
            and predict_hours_input is not None
            and end_timestamp_input is None
        ):
            start_timestamp = datetime.fromtimestamp(start_timestamp_input)
            predict_hours = int(predict_hours_input)
            end_timestamp = start_timestamp + timedelta(hours=predict_hours)

        elif (
            start_timestamp_input is not None
            and end_timestamp_input is not None
            and predict_hours_input is None
        ):
            start_timestamp = datetime.fromtimestamp(start_timestamp_input)
            end_timestamp = datetime.fromtimestamp(end_timestamp_input)
            predict_hours = int(
                (end_timestamp - start_timestamp).total_seconds() // 3600
            )

        elif (
            start_timestamp_input is not None
            and end_timestamp_input is not None
            and predict_hours_input is not None
        ):
            start_timestamp = datetime.fromtimestamp(start_timestamp_input)
            end_timestamp = datetime.fromtimestamp(end_timestamp_input)
            predict_hours = int(predict_hours_input)

            if (
                int((end_timestamp - start_timestamp).total_seconds() // 3600)
                != predict_hours
            ):
                raise HTTPException(
                    status_code=400,
                    detail="The difference between start and end timestamps does not match predict_hours.",
                )

        else:
            start_timestamp = current_hour
            end_timestamp = start_timestamp + timedelta(hours=24)
            predict_hours = 24

        start_timestamp_float = start_timestamp.timestamp()
        end_timestamp_float = end_timestamp.timestamp()

        # TODO: converting back and forward between timestamps and strings is not ideal
        timestamps = (
            pd.date_range(
                start=datetime.fromtimestamp(start_timestamp_float, tz=timezone),
                end=datetime.fromtimestamp(end_timestamp_float, tz=timezone),
                freq="h",
                tz=timezone,
            )
            .to_pydatetime()
            .tolist()[1:]
        )
        timestamps = [timestamp.strftime("%Y-%m-%dT%H:%M") for timestamp in timestamps]

        if len(timestamps) != predict_hours:
            raise HTTPException(
                status_code=400,
                detail="Invalid request, timestamps do not match predict_hours",
            )

        return start_timestamp_float, end_timestamp_float, predict_hours, timestamps

    async def _interpolate_prediction(
        self, prediction, prediction_shape, grid, lat, lon, timestamps
    ):
        if prediction_shape[1] == 1 and prediction_shape[2] == 1:
            prediction = prediction.squeeze(1).squeeze(1)

        elif prediction_shape[1] == 2 and prediction_shape[2] == 1:
            # Linear interpolation on latitude
            lat_grid = [grid[i, 0][0].item() for i in range(2)]

            lat_diff = lat_grid[1] - lat_grid[0]
            if abs(lat_diff) < 1e-9:  # Check for near-zero difference
                lat_weight = 0.5  # Default to equal weights if grid points are close
            else:
                lat_weight = (lat - lat_grid[0]) / lat_diff

            weights = (
                torch.tensor([1 - lat_weight, lat_weight]).unsqueeze(0).unsqueeze(2)
            )
            prediction = prediction * weights
            prediction = prediction.sum(dim=1).squeeze(1)

        elif prediction_shape[1] == 1 and prediction_shape[2] == 2:
            # Linear interpolation on longitude
            lon_grid = [grid[0, i][1].item() for i in range(2)]

            lon_diff = lon_grid[1] - lon_grid[0]
            if abs(lon_diff) < 1e-9:  # Check for near-zero difference
                lon_weight = 0.5  # Default to equal weights if grid points are close
            else:
                lon_weight = (lon - lon_grid[0]) / lon_diff

            weights = (
                torch.tensor([1 - lon_weight, lon_weight]).unsqueeze(0).unsqueeze(1)
            )
            prediction = prediction * weights
            prediction = prediction.sum(dim=2).squeeze(1)

        elif prediction_shape[1] == 2 and prediction_shape[2] == 2:
            # Bilinear interpolation
            lat_grid = [grid[i, 0][0].item() for i in range(2)]
            lon_grid = [grid[0, i][1].item() for i in range(2)]

            lat_diff = lat_grid[1] - lat_grid[0]
            lon_diff = lon_grid[1] - lon_grid[0]

            if abs(lat_diff) < 1e-9:
                lat_weight = 0.5
            else:
                lat_weight = (lat - lat_grid[0]) / lat_diff

            if abs(lon_diff) < 1e-9:
                lon_weight = 0.5
            else:
                lon_weight = (lon - lon_grid[0]) / lon_diff

            weights = torch.tensor(
                [
                    [
                        (1 - lat_weight) * (1 - lon_weight),
                        (1 - lat_weight) * lon_weight,
                    ],
                    [lat_weight * (1 - lon_weight), lat_weight * lon_weight],
                ]
            ).unsqueeze(0)

            weights = weights.expand(prediction.shape[0], 2, 2)
            prediction = prediction * weights
            prediction = prediction.sum(dim=1).sum(dim=1)

        else:
            logging.error(f"Invalid prediction shape: {prediction_shape}")
            raise HTTPException(status_code=500, detail="Invalid prediction shape")

        return {
            "prediction": prediction.tolist(),
            "grid": grid.tolist(),
            "timestamps": timestamps,
        }

    async def get_self(self):
        return self
