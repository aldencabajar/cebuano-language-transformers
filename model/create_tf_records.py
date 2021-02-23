import numpy as np 
import tensorflow as tf
import re
import sys
from tensorflow.data import Dataset
from tensorflow.sparse import to_dense as td
import tensorflow.keras as tfk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFGPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from tokenizers import Tokenizer
from tqdm import tqdm
import pandas as pd
import os

root_dir = os.getcwd()
sys.path.append('/'.join([root_dir, 'model']))
import fine_tuning as ft


def PreprocessData(df_txt, ids, tokenizer_lng):
        docs = df_txt.iloc[ids,:].document \
            .map(lambda str: str + '<|endoftext|>').tolist()
        pad_token_id = tokenizer_lng.token_to_id('<pad>')
        input = []
        labels = []
        attn_mask = []
        for doc in docs:
            encoded = tokenizer_lng.encode(doc)
            input.append(encoded.ids[:-1])
            labels.append(encoded.ids[1:])
            attn_mask.append(encoded.attention_mask[:-1])
        
        input = pad_sequences(input, value = pad_token_id, padding='post')
        labels = pad_sequences(labels, value = pad_token_id, padding='post')
        attn_mask = pad_sequences(attn_mask, value = 0, padding='post')

        return input, labels, attn_mask



def decode(serialized_example):
    # Decode examples stored in TFRecord
    # features = {'input': tf.io.FixedLenFeature([], tf.int64),
    #             'label': tf.io.FixedLenFeature([], tf.int64),
    #             'attn': tf.io.FixedLenFeature([], tf.int64)}

    features = {'input': tf.io.VarLenFeature(tf.int64),
                'label': tf.io.VarLenFeature(tf.int64),
                'attn': tf.io.VarLenFeature(tf.int64)}
    features = tf.io.parse_single_example(
        serialized_example,
        features=features
    )
 
    return features['input'], features['label'], features['attn']



 
def CreateTFRecord(df_text, tokenizer_lng, tfr_filename, ids):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    writer = tf.io.TFRecordWriter(tfr_filename)
    
    # iterate through all pkl files 
    for id_ in tqdm(ids):
        inp, lbl, attn = PreprocessData(df_txt, [id_], tokenizer_lng)
        feature = {'input': _int64_feature(inp.ravel()),
                    'label': _int64_feature(lbl.ravel()),
                    'attn':_int64_feature(attn.ravel())} 
    
        # Create an example protocol buffer
        example = tf.train.Example(
            features=tf.train.Features(feature=feature))
    
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()

if __name__ == '__main__':
    #path to save the TFRecords file
    train_tfr = 'model/train_oscar_corpus.tfrecords'
    test_tfr = 'model/test_oscar_corpus.tfrecords'
    path_to_tokenizer_lng = 'model/oscar-corpus-tokenizer.json'
    path_to_text = 'shuff-dedup/ceb/ceb_dedup.txt'
    tokenizer_lng = Tokenizer.from_file(path_to_tokenizer_lng)

    # open the file
    df_txt = ft.setup_examples(path_to_text)
    train_ids, test_ids = ft.train_test_split(df_txt.shape[0])

    CreateTFRecord(df_txt, tokenizer_lng, train_tfr, train_ids)
    CreateTFRecord(df_txt, tokenizer_lng, test_tfr, test_ids)

    emb_dim = tf.constant(len(tokenizer_lng.get_vocab()), dtype = tf.int64)

    train_dataset = tf.data.TFRecordDataset(train_tfr).map(decode) 
    train_dataset = train_dataset.map(lambda x, y, z: (td(x), td(y), td(z)))
    train_dataset = train_dataset.shuffle(10000).repeat()
    train_dataset = train_dataset.padded_batch(5)

    
    # for t in train_dataset.take(1):
    #     print(tf.sparse.to_dense(t[0]))

    list(train_dataset.take(1))


       

    # inp, lbl, attn = list(train_dataset.take(1))[0]
    # tf.sparse.to_dense(inp)


    





