from energonai.model import opt_125M, opt_30B, opt_175B, opt_6B
from typing import Any
from energonai import SubmitEntry
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List, Deque, Tuple, Hashable, Any
from energonai import BatchManager, SubmitEntry, TaskEntry
from transformers import OPTForCausalLM

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_model_fn():
    model_map = {
        'opt-125m': opt_125M, #(checkpoint='/data/yusun/xueyang/checkpoints/cs267/reshard-model_part-0.pt'),
        'opt-6.7b': opt_6B,
        'opt-30b': opt_30B,
        'opt-175b': opt_175B
    }
    return model_map['opt-125m']

class OptRequest(BaseModel):
    data: Any

def unpack_request(req: OptRequest) -> SubmitEntry: 
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
    return SubmitEntry(id(req), tokenizer(req.data))

# def tokenizer_func(req: OptRequest, tokenizer):
#     req.data = tokenizer(req.data, truncation=True, max_length=512)
#     return unpack_request(req)
    

class OptResponse(BaseModel):
    output: Any

def pack_response(output: Any) -> OptResponse:

    return OptResponse(output=output)


class BatchManagerForGeneration(BatchManager):
    def __init__(self, max_batch_size: int = 1, pad_token_id: int = 0) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id

    def _left_padding(self, batch_inputs):
        max_len = max(len(inputs['input_ids']) for inputs in batch_inputs)
        outputs = {'input_ids': [], 'attention_mask': []}
        for inputs in batch_inputs:
            input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            padding_len = max_len - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            outputs['input_ids'].append(input_ids)
            outputs['attention_mask'].append(attention_mask)
        for k in outputs:
            outputs[k] = torch.tensor(outputs[k])
        return outputs, max_len

    @staticmethod
    def _make_batch_key(entry: SubmitEntry) -> tuple:
        data = entry.data
        return (data['top_k'], data['top_p'], data['temperature'])

    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()
        uids = [entry.uid]
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            if self._make_batch_key(entry) != self._make_batch_key(q[0]):
                break
            if q[0].data['max_tokens'] > entry.data['max_tokens']:
                break
            e = q.popleft()
            batch.append(e.data)
            uids.append(e.uid)
        inputs, max_len = self._left_padding(batch)
        trunc_lens = []
        for data in batch:
            trunc_lens.append(max_len + data['max_tokens'])
        inputs['top_k'] = entry.data['top_k']
        inputs['top_p'] = entry.data['top_p']
        inputs['temperature'] = entry.data['temperature']
        inputs['max_tokens'] = max_len + entry.data['max_tokens']
        return TaskEntry(tuple(uids), inputs), {'trunc_lens': trunc_lens}

    def split_batch(self, task_entry: TaskEntry, trunc_lens: List[int] = []) -> List[Tuple[Hashable, Any]]:
        retval = []
        for uid, output, trunc_len in zip(task_entry.uids, task_entry.batch, trunc_lens):
            retval.append((uid, output[:trunc_len]))
        return retval