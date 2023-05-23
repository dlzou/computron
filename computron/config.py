from collections.abc import Callable
from dataclasses import dataclass, field

from computron.batch_manager import BatchManager


@dataclass
class EngineConfig:
    master_host: str
    master_port: int
    rpc_port: int
    max_loaded: int
    pipe_size: int = 1
    queue_size: int = 0
    rpc_disable_shm: bool = True


@dataclass
class ModelConfig:
    model_id: str
    model_fn: Callable
    model_kwargs: dict[str, object] = field(default_factory=dict) 
    pipelinable: bool = False
    batch_manager: BatchManager | None = None
