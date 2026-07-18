# -*- coding: utf-8 -*-


import gc
import logging
import math
import json
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Tuple, Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, is_torch_npu_available
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from vllm.inputs.data import TokensPrompt
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    CrossEncoder = object  # type: ignore
    SentenceTransformer = object  # type: ignore

try:
    from langchain_core.documents import Document
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

try:
    from torch.utils.data._utils.worker import ManagerWatchdog
except ImportError:
    pass


class Qwen3ReRankervLLM(CrossEncoder):
    def __init__(self, model_path, instruction="Given the user query, retrieval the relevant passages", **kwargs):
        number_of_gpu=torch.cuda.device_count()
        self.instruction = instruction
        # 添加 local_files_only=True 强制从本地路径加载，避免被误认为是 Hugging Face Hub repo id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_length=kwargs.get('max_length', 8192)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(temperature=0,
            top_p=0.95,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token,self.false_token],
        )
        self.lm = LLM(
            model=model_path,
            tensor_parallel_size=number_of_gpu,
            max_model_len=10000,
            enable_prefix_caching=True,
            distributed_executor_backend='ray',
            gpu_memory_utilization=0.8
        )


    def format_instruction(self, instruction, query, doc):
        if isinstance(query, tuple):
            instruction = query[0]
            query = query[1]
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text


    def rank(self, query, candidate_docs, topk=10):
        queries = [query] * len(candidate_docs)
        messages = [self.format_instruction(self.instruction, query, doc.page_content) for query, doc in zip(queries, candidate_docs)]
        messages =  self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:self.max_length] + self.suffix_tokens for ele in messages]
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        outputs = self.lm.generate(messages, self.sampling_params, use_tqdm=False)
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            token_count = len(outputs[i].outputs[0].token_ids)
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token].logprob
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)

        response = [
            doc
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )
            ][:topk]
        return response


    def stop(self):
        destroy_model_parallel()


if __name__ == '__main__':
    qwen3_reranker = "./models/Qwen3-Reranker-4B"
    bge_rerank = Qwen3ReRankervLLM(qwen3_reranker)
    query = "今天天气怎么样"
    docs = ["你好", "今天天气不错", "今天有雨吗"]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = bge_rerank.rank(query, docs)
    print(response)
