from dataclasses import dataclass
from typing import Hashable, Tuple


@dataclass
class OffloadSignal:
    uids: Tuple[Hashable, ...]
    on_device: bool
