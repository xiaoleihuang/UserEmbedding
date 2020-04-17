'''Average BERT outputs as user representations.
'''

import torch
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification
