from huggingface_DataS import load_dataset
from transformers import T5Tokenizer

model_name = 'unicamp-dl/ptt5-base-portuguese-vocab'

tokenizer = T5Tokenizer.from_pretrained(model_name)

dataset = load_dataset("yelp_review_full")
dataset["train"][100]