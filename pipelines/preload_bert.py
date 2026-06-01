from transformers import AutoTokenizer, BertModel
from pathlib import Path
text_encoder_type = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
tokenizer.save_pretrained(str(Path("checkpoints") / text_encoder_type))
bertmodel = BertModel.from_pretrained(text_encoder_type)
bertmodel.save_pretrained(str(Path("checkpoints") / text_encoder_type))
