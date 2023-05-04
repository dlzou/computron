from typing import List, Deque, Tuple, Hashable, Any, Union, Optional

from colossalai.logging import get_dist_logger
from computron import OffloadEntry, OffloadingBatchManager
from energonai import BatchManager, SubmitEntry, TaskEntry
from energonai.model import opt_125M, opt_1B, opt_6B, opt_30B, opt_175B
from pydantic import BaseModel, Field
import torch
from transformers import GPT2Tokenizer


logger = get_dist_logger("colossalai")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-30b")


def get_model_fn(model_name: str):
    model_map = {
        "opt-125m": opt_125M,  # (checkpoint='/data/yusun/xueyang/checkpoints/cs267/reshard-model_part-0.pt'),
        "opt-1.3b": opt_1B,
        "opt-6.7b": opt_6B,
        "opt-30b": opt_30B,
        "opt-175b": opt_175B,
    }
    return model_map[model_name]


class OPTRequest(BaseModel):
    max_tokens: int = Field(gt=0, le=256, default=64)
    prompt: str = Field(
        min_length=1,
        default="Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:",
    )
    top_k: Optional[int] = Field(default=50, gt=0, example=50)
    top_p: Optional[float] = Field(default=0.5, gt=0.0, lt=1.0, example=0.5)
    temperature: Optional[float] = Field(default=0.7, gt=0.0, lt=1.0, example=0.7)


class OPTResponse(BaseModel):
    output: Any


def unpack_request(req: OPTRequest) -> SubmitEntry:
    inputs = tokenizer(req.prompt, truncation=True, max_length=512)
    # For now, only support single token generation due to complications with PP.
    # TODO: proper iteration-level scheduling
    inputs["max_tokens"] = 1  # req.max_tokens
    inputs["top_k"] = req.top_k
    inputs["top_p"] = req.top_p
    inputs["temperature"] = req.temperature
    return SubmitEntry(id(req), inputs)


def pack_response(output: Any) -> OPTResponse:
    output = tokenizer.decode(output, skip_special_tokens=True)
    return OPTResponse(output=output)


class OPTBatchManager(OffloadingBatchManager):
    def __init__(self, max_batch_size: int = 1, pad_token_id: int = 0) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id

    def _left_padding(self, batch_inputs):
        max_len = max(len(inputs["input_ids"]) for inputs in batch_inputs)
        outputs = {"input_ids": [], "attention_mask": []}
        for inputs in batch_inputs:
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
            padding_len = max_len - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            outputs["input_ids"].append(input_ids)
            outputs["attention_mask"].append(attention_mask)
        for k in outputs:
            outputs[k] = torch.tensor(outputs[k])
        return outputs, max_len

    @staticmethod
    def _make_batch_key(entry: SubmitEntry) -> tuple:
        data = entry.data
        return (data["top_k"], data["top_p"], data["temperature"])

    def make_batch(
        self, q: Deque[Union[SubmitEntry, OffloadEntry]]
    ) -> Tuple[Union[TaskEntry, OffloadEntry], dict]:
        entry = q.popleft()
        if isinstance(entry, OffloadEntry):
            return entry, {}
        uids = [entry.uid]
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            if isinstance(q[0], OffloadEntry):
                break
            if self._make_batch_key(entry) != self._make_batch_key(q[0]):
                break
            if q[0].data["max_tokens"] > entry.data["max_tokens"]:
                break
            e = q.popleft()
            batch.append(e.data)
            uids.append(e.uid)
        inputs, max_len = self._left_padding(batch)
        trunc_lens = []
        for data in batch:
            trunc_lens.append(max_len + data["max_tokens"])
        inputs["top_k"] = entry.data["top_k"]
        inputs["top_p"] = entry.data["top_p"]
        inputs["temperature"] = entry.data["temperature"]
        inputs["max_tokens"] = max_len + entry.data["max_tokens"]
        return TaskEntry(tuple(uids), inputs), {"trunc_lens": trunc_lens}

    def split_batch(
        self, task_entry: TaskEntry, trunc_lens: List[int] = []
    ) -> List[Tuple[Hashable, Any]]:
        outputs = task_entry.batch["input_ids"]
        retval = []
        for uid, output, trunc_len in zip(task_entry.uids, outputs, trunc_lens):
            retval.append((uid, output[:trunc_len]))
        return retval


class BatchManagerForGeneration(BatchManager):
    def __init__(self, max_batch_size: int = 1, pad_token_id: int = 0) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id

    def _left_padding(self, batch_inputs):
        max_len = max(len(inputs["input_ids"]) for inputs in batch_inputs)
        outputs = {"input_ids": [], "attention_mask": []}
        for inputs in batch_inputs:
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
            padding_len = max_len - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            outputs["input_ids"].append(input_ids)
            outputs["attention_mask"].append(attention_mask)
        for k in outputs:
            outputs[k] = torch.tensor(outputs[k])
        return outputs, max_len

    @staticmethod
    def _make_batch_key(entry: SubmitEntry) -> tuple:
        data = entry.data
        return (data["top_k"], data["top_p"], data["temperature"])

    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()
        uids = [entry.uid]
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            if self._make_batch_key(entry) != self._make_batch_key(q[0]):
                break
            if q[0].data["max_tokens"] > entry.data["max_tokens"]:
                break
            e = q.popleft()
            batch.append(e.data)
            uids.append(e.uid)
        inputs, max_len = self._left_padding(batch)
        trunc_lens = []
        for data in batch:
            trunc_lens.append(max_len + data["max_tokens"])
        inputs["top_k"] = entry.data["top_k"]
        inputs["top_p"] = entry.data["top_p"]
        inputs["temperature"] = entry.data["temperature"]
        inputs["max_tokens"] = max_len + entry.data["max_tokens"]
        return TaskEntry(tuple(uids), inputs), {"trunc_lens": trunc_lens}

    def split_batch(
        self, task_entry: TaskEntry, trunc_lens: List[int] = []
    ) -> List[Tuple[Hashable, Any]]:
        outputs = task_entry.batch["input_ids"]
        retval = []
        for uid, output, trunc_len in zip(task_entry.uids, outputs, trunc_lens):
            retval.append((uid, output[:trunc_len]))
        return retval
