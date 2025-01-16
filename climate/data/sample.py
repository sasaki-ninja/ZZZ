from dataclasses import dataclass
import torch

from climate.protocol import TimePredictionSynapse

@dataclass
class Era5Sample:
    start_timestamp: float
    end_timestamp: float
    input_data: torch.Tensor
    output_data: torch.Tensor

    @property
    def predict_hours(self) -> int:
        """
        Returns the number of hours the miner should predict.
        """
        return self.output_data.shape[0]

    def get_synapse(self) -> TimePredictionSynapse:
        """
        Converts the sample to a TimePredictionSynapse which miners can predict on.
        Note that the output data is NOT set in this synapse.
        """
        return TimePredictionSynapse(
            input_data=self.input_data.tolist(), 
            requested_hours=self.predict_hours,
        )
