from app.train.models.bert import Model as BertClassifier
from app.train.models.bert_tiny import Model as BertTinyClassifier
# Legacy aliases
Model = BertClassifier
BertTinyModel = BertTinyClassifier
