from typing import Optional
import pandas as pd

def timestamp_to_str(float_timestamp: float) -> str:
    return get_timestamp(float_timestamp).strftime("%Y-%m-%d %H:%M:%S")

def get_today(floor: Optional[str] = None) -> pd.Timestamp:
    """
    Copernicus is inside GMT+0, so we can always use that timezone to get the current day and hour matching theirs.
    But then remove the timezone information so we can actually compare with the dataset (which is TZ-naive).
    """

    timestamp = pd.Timestamp.now(tz = 'GMT+0').replace(tzinfo=None)
    if floor:
        return timestamp.floor(floor)
    return timestamp

def get_timestamp(float_timestamp: float) -> pd.Timestamp:
    """
    Convert a float timestamp (used for storage) to a pandas timestamp, considering that Copernicus is inside GMT+0.
    We strip off the timezone information to make it TZ-naive again (but according to Copernicus' time).
    """
    return pd.Timestamp(float_timestamp, unit="s", tz='GMT+0').replace(tzinfo=None)
    