import numpy as np 
import tensorflow as tf
import re
import sys
from tensorflow.data import Dataset
import tensorflow.keras as tfk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFGPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from tokenizers import Tokenizer
from tqdm import tqdm
import pandas as pd
import os


"""
#initializing tpu
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
"""

def setup_model_finetuning(path_to_pretrained, tokenizer_en, tokenizer_lng):
    # load pre-trained models
    model = TFGPT2LMHeadModel.from_pretrained(path_to_pretrained)

    # setup new embedding matrix for fine-tuning 
    weights = tf.stop_gradient(
        model.transformer.get_input_embeddings() \
        .weight.value()) \
        .numpy()

    # get mean embeddings
    mean_weights = tf.reduce_mean(weights, axis = 0).numpy()

    new_vocab = tokenizer_lng.get_vocab()
    old_vocab = tokenizer_en.get_vocab()
    new_embeddings = tf.zeros([
        len(new_vocab), mean_weights.shape[0]
    ]).numpy()
    
    for word, idx_new in new_vocab.items():
        idx_old =  old_vocab.get(word, -1)
        if idx_old >= 0:
            new_embeddings[idx_new, :] = weights[idx_old, :]
        else:
            new_embeddings[idx_new, :] = mean_weights

    # set embeddings
    model.transformer.set_input_embeddings(
        tf.constant(new_embeddings)
    )

    # freezing model weights 
    for layer in model.transformer.h:
        layer.trainable = False

        model.transformer.wte.trainable = True
        model.transformer.wpe.trainable = True
        model.transformer.ln_f.trainable = True

    return model


def setup_examples(path_to_txt):
    text_lines = []
    ptt = re.compile(r'\n$|\[\d+\]')
    with open(path_to_txt, 'r') as f:
        for line in tqdm(f):
            clean_text = ptt.sub('', line)
            text_lines.append(clean_text)
    df_txt = pd.DataFrame({'document': text_lines})

    # add end of line token
    df_txt['truncated_doc'] = df_txt.document \
    .apply(lambda x: x + '<|endoftext|>')

    return df_txt

def CreateDataset(ids, df_txt, tokenizer_lng, batch_size):
    tokenizer_lng.enable_truncation(max_length=1024)
    def PreprocessData(ids):
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

    # prepare tf datasets 
    def _dataset(ids, batch_size = None):
        df = Dataset.from_tensor_slices(ids)
        df = df.shuffle(len(ids)).batch(batch_size)
        df = df.map(
            lambda x: tf.py_function(PreprocessData, [x], 
            [tf.int32, tf.int32, tf.int32])
        )
        return df

    dataset = _dataset(ids, batch_size=batch_size) 
    return dataset


def train_test_split(num_examples, train_split = 0.8):
    train_num_docs = int(num_examples * train_split)
    train_ids = np.random.choice(
        num_examples,
        train_num_docs,
        replace = False
    )
    test_ids = np.setdiff1d(np.arange(num_examples), train_ids)
    return train_ids, test_ids

    
    

    # preprocessing routine 
@tf.function
def train_step(model, iterator):
  """The step function for one training step"""

  def step_fn(inputs):
    """The computation to run on each TPU device."""
    inp, labels, attn = inputs
    with tf.GradientTape() as tape:
      logits = model(input_ids= inp, attention_mask = attn)
      loss = tfk.losses.sparse_categorical_crossentropy(
          labels, logits, from_logits=True)
      loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    training_loss.update_state(loss * strategy.num_replicas_in_sync)
    training_accuracy.update_state(labels, logits)

  strategy.run(step_fn, args=(next(iterator),))





            
if __name__ == '__main__':

    # parameters 
    path_to_tokenizer_en = 'model/en_tokenizer'
    path_to_tokenizer_lng = 'model/oscar-corpus-tokenizer.json'
    path_to_txt = 'shuff-dedup/ceb/ceb_dedup.txt'
    path_to_pretrained = 'model/en_pretrained_gpt2'
    train_split = 0.8
    train_batch_size = 32 
    test_batch_size = 1
    epochs = 1
    steps = None 
    lr = int(1e-4)
    log = False
    device = 'gpu'

    if device == 'tpu':
        #initializing tpu
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))

        strategy = tf.distribute.TPUStrategy(resolver)

        

    # load models
    tokenizer_en = GPT2TokenizerFast.from_pretrained(path_to_tokenizer_en)
    tokenizer_lng = Tokenizer.from_file(path_to_tokenizer_lng)

    # create train and test dataset
    df_txt = setup_examples(path_to_txt)
    train_ids, test_ids = train_test_split(df_txt.shape[0])

    if device == 'tpu':
        with strategy.scope():
            model = setup_model_finetuning(path_to_pretrained, tokenizer_en, tokenizer_lng)
            optimizer = tf.keras.optimizers.Adam()
            training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
            training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                'training_accuracy', dtype=tf.float32)

        # Calculate per replica batch size, and distribute the datasets on each TPU
        # worker.
        per_replica_batch_size = train_batch_size // strategy.num_replicas_in_sync

        train_dataset = strategy.experimental_distribute_datasets_from_function(
            lambda _: CreateDataset(train_ids, df_txt, tokenizer_lng, batch_size = per_replica_batch_size))

    else:
        train_dataset = CreateDataset(train_ids, df_txt, tokenizer_lng, batch_size=train_batch_size)
    
    inp, lbl, attn = list(train_dataset.take(1))[0]
    print(inp)






"""
# Training routine ==============================================
# evaluation metrics
acc = tfk.metrics.SparseCategoricalAccuracy()
def cross_entropy_loss(y_true,y_pred):
    return(tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)))

if steps is None:
    steps = train_ids.
    shape[0] // train_batch_size
optimizer = tfk.optimizers.Adam(learning_rate = lr)


for i in range(epochs):
    pbar = tqdm(df_train.take(steps))
    for j, (inp, label, attn) in enumerate(pbar):
        batch_losses = []
        with tf.GradientTape() as tape:
            results = model(input_ids = inp, attention_mask = label)
            loss = cross_entropy_loss(label, results.logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # get accuracy
        acc.update_state(label, results.logits)

        pbar.set_description(f"Epoch {i + 1}, batch {j + 1}")
        pbar.set_postfix({'cross-entropy loss': loss.numpy(), 
                        'accuracy':acc.result().numpy()})
        acc.reset_states()
        """

