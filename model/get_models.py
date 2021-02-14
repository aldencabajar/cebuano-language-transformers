import tensorflow as tf
import os
from transformers import TFGPT2LMHeadModel,GPT2TokenizerFast
path_to_en_tokenizer = 'model/en_tokenizer'

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

tokenizer.save_pretrained(path_to_en_tokenizer)
model.save_pretrained('model/')





