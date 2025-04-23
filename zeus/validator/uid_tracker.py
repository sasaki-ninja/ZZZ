from typing import List
import time
import threading
import bittensor as bt
from zeus.base.validator import BaseValidatorNeuron
from zeus.utils.uids import get_random_uids
from zeus.validator.constants import MAINNET_UID

class UIDTracker:

    def __init__(self, validator: BaseValidatorNeuron):
        self.validator = validator
        self._busy_uids = set()
        self._last_good_uids = set()
        self.lock = threading.Lock()

    def get_random_uids(self, k: int, tries: int = 3, sleep: int = 1) -> List[int]:
        attempt = 1
        while True:
            if attempt > 1:
                 # sleep here so no delay once we sample miners to add them to our busy-list
                 bt.logging.warning(f"Failed to sample enough non-busy miner uids, retrying in {sleep} second(s). ATTEMPT {attempt}/{tries}")
                 time.sleep(sleep)

            miner_uids = get_random_uids(
                self.validator.metagraph,
                k,
                self.validator.config.neuron.vpermit_tao_limit,
                MAINNET_UID,
                # on last attempt, we ignore busyness to ensure we are able to query
                exclude=self.get_busy_uids() if attempt < tries else set()
            )
            if len(miner_uids) == k or attempt == tries:
                self.add_busy_uids(miner_uids)
                return miner_uids
            attempt += 1
    
    # NOTE: since this is access by proxy and forward, need a thread lock!
    def get_busy_uids(self) -> List[int]:
        with self.lock:
            return self._busy_uids
    
    def add_busy_uids(self, uids: List[int]):
        with self.lock:
            self._busy_uids.update(set(uids))

    def mark_finished(self, uids: List[int], good: bool = False):
        with self.lock:
            self._busy_uids = self._busy_uids - set(uids)
            if good:
                self._last_good_uids = set(uids)

    def get_responding_uids(self, k: int) -> List[int]:
        with self.lock:
            self._last_good_uids = self._last_good_uids - self._busy_uids
        responding_uids = list(self._last_good_uids)[:k]
        self.add_busy_uids(responding_uids)
        return responding_uids
