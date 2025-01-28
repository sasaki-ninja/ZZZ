from typing import Optional, Tuple
import torch

from zeus.utils.coordinates import get_bbox
from zeus.protocol import TimePredictionSynapse

class Era5Sample:

    def __init__(
            self,
            start_timestamp:float,
            end_timestamp: float,
            lat_start: Optional[float] = None,
            lat_end: Optional[float] = None,
            lon_start: Optional[float] = None,
            lon_end: Optional[float] = None,
            input_data: Optional[torch.Tensor] = None,
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

        self.lat_start, self.lat_end = lat_start, lat_end
        self.lon_start, self.lon_end = lon_start, lon_end

        self.input_data = input_data 
        self.output_data = output_data
        self.predict_hours = predict_hours

        if input_data is not None:
            self.lat_start, self.lat_end, self.lon_start, self.lon_end = get_bbox(input_data)
        elif lat_start is None or lat_end is None or lon_start is None or lon_end is None:
            raise ValueError("Either input data or lat/lon ranges must be provided.")

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
            input_data=self.input_data.tolist(), 
            requested_hours=self.predict_hours,
        )
