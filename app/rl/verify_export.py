"""RL 模型导出验证 — 确认合并模型加载 + 推理正常。

Ported from XIAOMI_SU7_RAG/src/rl/verify_export.py。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Verify RL model export")
    parser.add_argument("--model", default="models/qwen3_lora_rl", help="Exported model path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        test_query = "小米SU7的续航里程是多少？"
        system_prompt = "你是小米SU7车型的专业问答助手。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_query},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.3, do_sample=True)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        logger.info(f"Query: {test_query}")
        logger.info(f"Response: {response}")

        has_answer = "<answer>" in response
        has_search = "<search_local>" in response or "<search_web>" in response

        logger.info(f"Tags check: answer={has_answer}, search={has_search}")
        if has_answer:
            logger.info("✅ Export verification passed")
        else:
            logger.warning("⚠️ No <answer> tag in response — model may need more training")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
