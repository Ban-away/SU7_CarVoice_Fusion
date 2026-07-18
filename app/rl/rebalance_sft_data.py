"""RL 训练数据再平衡 — 控制 web/local 轨迹比例。

Ported from XIAOMI_SU7_RAG/src/rl/rebalance_sft_data.py。

默认：保留全部 web 轨迹，下采样 local 轨迹，使 web 占比达到 ~33%。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = "data/training/rl"


def main():
    parser = argparse.ArgumentParser(description="Rebalance RL training data")
    parser.add_argument("--local-cap", type=int, default=1000, help="Local trajectory cap")
    parser.add_argument("--restore", action="store_true", help="Restore from backups")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    data_dir = Path(DATA_DIR)
    for fname in ["combined_trajectories_sft.json", "combined_trajectories_grpo.jsonl"]:
        path = data_dir / fname
        if not path.exists():
            logger.info(f"File not found: {path}")
            continue

        if args.restore:
            backup = data_dir / f"{fname}.original"
            if backup.exists():
                shutil.copy(backup, path)
                logger.info(f"Restored {fname} from backup")
            continue

        # Backup original
        backup = data_dir / f"{fname}.original"
        shutil.copy(path, backup)

        if fname.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))

        web_items = [d for d in data if d.get("data_source") == "web_fallback"]
        local_items = [d for d in data if d.get("data_source") != "web_fallback"]
        logger.info(f"Before: {len(web_items)} web + {len(local_items)} local = {len(data)} total")

        # Keep all web, cap local
        if len(local_items) > args.local_cap:
            random.seed(42)
            local_items = random.sample(local_items, args.local_cap)

        balanced = web_items + local_items
        random.shuffle(balanced)

        if fname.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(balanced, f, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                for item in balanced:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        web_pct = len(web_items) / len(balanced) * 100 if balanced else 0
        logger.info(f"After:  {len(web_items)} web ({web_pct:.0f}%) + {len(local_items)} local = {len(balanced)} total")


if __name__ == "__main__":
    main()
