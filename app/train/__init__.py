"""CarVoice BERT 分类模型训练框架。

Ported from CarVoice_Agent/train/。
使用 HuggingFace transformers 替代自定义 BERT 实现。

训练:
  python -m app.train.run --model bert --data intent     # 意图分类
  python -m app.train.run --model bert_tiny --data reject # 拒识分类

推理服务:
  python app/train/intent_server.py   # 意图服务 (8008)
  python app/train/reject_server.py   # 拒识服务 (8007)
  python app/train/nlu_server.py      # NLU 服务 (8009)
"""
