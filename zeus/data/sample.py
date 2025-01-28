from typing import Optional, Tuple
import torch

from zeus.utils.coordinates import get_bbox
from zeus.protocol import TimePredictionSynapse

class Era5Sample:

    def __init__(
            self,
            start_timestamp:float,
            end_timestamp: float,
            lat_start: float,
            lat_end: float,
            lon_start: float,
            lon_end: float,
            output_data: Optional[torch.Tensor] = None,
            predict_hours: Optional[int] = None
    ):
        """
        Create a datasample, either containing actual data or representing a database entry.

        Note that if input_data is provided, the lat/lon ranges are automatically overwritten.
        Note that if output_data is provided, the predict_hours is automatically overwritten.
        """
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        self.lat_start = lat_start
        self.lat_end = lat_end
        self.lon_start = lon_start
        self.lon_end = lon_end

        self.output_data = output_data
        self.predict_hours = predict_hours

        if output_data is not None:
            self.predict_hours = output_data.shape[0]
        elif predict_hours is None:
            raise ValueError("Either output data or predict hours must be provided.")

    def get_bbox(self) -> Tuple[float]:
        return self.lat_start, self.lat_end, self.lon_start, self.lon_end

    def get_synapse(self) -> TimePredictionSynapse:
        """
        Converts the sample to a TimePredictionSynapse which miners can predict on.
        Note that the output data is NOT set in this synapse.
        """
        return TimePredictionSynapse(
            bounding_box=self.get_bbox(),
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            requested_hours=self.predict_hours
        )
