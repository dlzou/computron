import os
from typing import Any, Deque, Hashable, List, Tuple, Union

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.utils import is_using_pp
from computron import LoadEntry, OffloadingBatchManager
from energonai import SubmitEntry, TaskEntry
from pydantic import BaseModel
from titans.model.vit.vit import _create_vit_model
import torch


class DummyDataloader:
    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size

    def generate(self):
        data = torch.rand(self.batch_size, 3, 224, 224)
        label = torch.randint(low=0, high=10, size=(self.batch_size,))
        return data, label

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length


def create_vit():
    # launch from torch
    # parser = colossalai.get_default_parser()
    # args = parser.parse_args()
    # colossalai.launch_from_torch(config=args.config)

    # model config
    # img_size=224, patch_size=16, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4
    IMG_SIZE = 224
    PATCH_SIZE = 32
    HIDDEN_SIZE = 1024
    DEPTH = 24
    NUM_HEADS = 16
    MLP_RATIO = 4
    NUM_CLASSES = 10
    CHECKPOINT = False
    SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE) ** 2 + 1  # add 1 for cls token

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    if hasattr(gpc.config, "LOG_PATH"):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    use_pipeline = is_using_pp()

    # create model
    model_kwargs = dict(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        # vit_ratio=MLP_RATIO,
        num_classes=10,
        init_method="jax",
        checkpoint=CHECKPOINT,
    )

    if use_pipeline:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = _create_vit_model(**model_kwargs)
        pipelinable.to_layer_list()
        pipelinable.policy = "uniform"
        model = pipelinable.partition(
            1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE)
        )
    else:
        model = _create_vit_model(**model_kwargs)

    # count number of parameters
    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_stage = 0
    else:
        pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    logger.info(f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")

    return model


class ViTRequest(BaseModel):
    data: Any


class ViTResponse(BaseModel):
    output: Any


def unpack_request(req: ViTRequest) -> SubmitEntry:
    return SubmitEntry(id(req), req.data)


def pack_response(output: Any) -> ViTResponse:
    return ViTResponse(output=output)


class ViTBatchManager(OffloadingBatchManager):
    def __init__(self, max_batch_size: int = 1):
        self.max_batch_size = max_batch_size

    def make_batch(
        self, q: Deque[Union[SubmitEntry, LoadEntry]]
    ) -> Tuple[Union[TaskEntry, LoadEntry], dict]:
        entry = q.popleft()
        if isinstance(entry, LoadEntry):
            return entry, {}

        uids = [entry.uid]
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            if isinstance(q[0], LoadEntry):
                break
            entry = q.popleft()
            uids.append(entry.uid)
            batch.append(entry.data)
        inputs = torch.stack(batch)
        return TaskEntry(tuple(uids), inputs), {}

    def split_batch(self, task_entry: TaskEntry) -> List[Tuple[Hashable, Any]]:
        ret = []
        for uid, output in zip(task_entry.uids, task_entry.batch):
            ret.append((uid, output))
        return ret
