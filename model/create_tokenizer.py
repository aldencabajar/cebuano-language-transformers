import json
import os
import sys
import pandas as pd
import numpy as np
import re
from tokenizers import normalizers, TextInputSequence, Tokenizer
from tokenizers.normalizers import NFD, NFKC, StripAccents, Lowercase, Sequence
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


def load_text_file_json(file, text_attr):
    """
    Loads text line by line from a json file 
    """
    text = []
    for line in open(file, 'r'):
        json_obj = json.loads(line)
        text.append(json_obj[text_attr])
    return text

def load_text_file(file):
    text = []
    with open(file, 'r') as f:
        for line in f:
            text.append(line)
    return text

def write_text_file(text, path):
    with open(path, 'w') as f:
        for line in text:
            _txt = re.sub(r'^(.*)\n\n*', '', line)
            # remove redundant new lines
            _txt = re.sub(r'\n(?!$)', '', _txt)
            _txt = re.sub(r'\n\n', '\n', _txt)
            f.write(_txt)
        f.close()

def tokenizer_pipeline():
    """
    specific pipeline for Cebuano Corpus tokenization 
    - Uses a Byte pair encoding (BPE) tokenizer
    """
    tokenizer = Tokenizer(BPE())

    # string normalization
    tokenizer.normalizer = Sequence([
        NFD(),
        StripAccents(),
        Lowercase()
    ])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer

if __name__ == "__main__":
    # preparing corpus for wiki 
    en_vocab_size =50257
    wiki_txt = load_text_file_json('text/AA/wiki_00.json', 'text')
    write_text_file(wiki_txt, 'wiki-corpus.txt')

    corpus_files = {
        'wiki-corpus': 'wiki-corpus.txt',
        'oscar-corpus': 'shuff-dedup/ceb/ceb_dedup.txt'
    } 

    # define a trainer for the tokenizer
    trainer = BpeTrainer(vocab_size=en_vocab_size, show_progress=True, 
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens=["<|endoftext|>"])

    for corpus, path in corpus_files.items():
        tokenizer = tokenizer_pipeline()
        tokenizer.train([path], trainer)
        tokenizer.save(f'model/{corpus}-tokenizer.json')







