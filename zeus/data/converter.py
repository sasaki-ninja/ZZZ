from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np

@dataclass
class VariableConverter(ABC):
    """
    Utilities to convert OpenMeteo's variables and units to normal ERA5 
    and vice versa
    """

    # their main ERA5 representation, used as key throughout
    data_var: str 

    # OpenMeteo variable name
    om_name: str
    # Abbreviated ERA5 name, which is how they are saved internally in NC files
    short_code: str
    # Metric SI unit as string
    unit: str

    def era5_to_om(self, data: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return data

    def om_to_era5(self, data: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return data
    

class TemperatureConverter(VariableConverter):

    def era5_to_om(self, data: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return data - 273.15
    
    def om_to_era5(self, data: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return data + 273.15


class PrecipitationConverter(VariableConverter):

    def era5_to_om(self, data: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return data * 1e3
    
    def om_to_era5(self, data: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return data / 1e3
    
    
REGISTRY = {converter.data_var: converter for converter in [
        TemperatureConverter("2m_temperature", om_name="temperature_2m", short_code="t2m", unit="K"), 
        PrecipitationConverter("total_precipitation", om_name="precipitation", short_code="tp", unit="m/h"),
]}

def get_converter(data_var: str) -> VariableConverter:
    try:
        return REGISTRY[data_var]
    except KeyError:
        raise NotImplementedError(f"Variable {data_var} does not exist in registry")
