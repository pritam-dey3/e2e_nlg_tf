import tensorflow as tf
import numpy as np

import argparse
from transformers import AutoTokenizer

from e2e_transformers.model import E2ETransformer
from e2e_transformers.lr_scheduler import CustomSchedule
from e2e_transformers.utils import create_masks
from data_preprocessing import preprocessing, preprocessing_py_func

parser = argparse.ArgumentParser()


parser.add_argument('-train_path', default=None)    
parser.add_argument('-val_path', default='val_data.csv')    

parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-d_inner_hid', type=int, default=2048)

parser.add_argument('-embedding', type=str, default='t5_extended_embed.npy')

parser.add_argument('-n_heads', type=int, default=8)
parser.add_argument('-n_enc_layers', type=int, default=3)
parser.add_argument('-n_dec_layers', type=int, default=6)
parser.add_argument('-max_len', type=int, default=100)

parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
parser.add_argument('-pad_idx', type=int, default=0)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-buffer', type=int, default=20000)


opt = parser.parse_args()

opt.n_heads = 2
opt.n_dec_layers = 3
opt.embedding = "dummy_embed.npy"
opt.d_inner_hid = 64
opt.b = 32
opt.epoch = 2

# python train.py -train_path cleaned-data/test-fixed.csv -epoch 2 -b 32 -d_inner_hid=64 -embedding dummy_embed.npy -n_heads 2 -n_dec_layers 3
print(opt)

e2e_model = E2ETransformer.from_config(opt)
# e2e_model.load_weights("./saved_model/e2e")
# print(e2e_model.opt)

embedding_weight = np.load(opt.embedding)
vocab_size, d_model = embedding_weight.shape

learning_rate = CustomSchedule(d_model)


optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                    epsilon=1e-9)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(e2e_model = e2e_model,
                        optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print ('Latest checkpoint restored!!')

print(e2e_model.opt)


tokenizer = AutoTokenizer.from_pretrained("t5-small")

special_tokens = {'additional_special_tokens': ['<area>', '<eatType>', '<food>', '<near>',                                                      '<name>', '<customer rating>', '<priceRange>',                                                   '<familyFriendly>', '<notfamilyFriendly>',                                                       '<cr_slot>', '<pr_slot>', '<sos>']}
tokenizer.add_special_tokens(special_tokens)


def evaluate(inp, slot_inp):

    # inp sentence is portuguese, hence adding the start and end token

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer.convert_tokens_to_ids('<sos>')]
    output = tf.expand_dims(decoder_input, 0)
        
    for i in range(100):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = e2e_model(inp, slot_inp, output, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer.convert_tokens_to_ids('</s>'):
            return tf.squeeze(output, axis=0), attention_weights
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(inp, slot_inp, text):
    result, attention_weights = evaluate(inp, slot_inp)
    
    predicted_sentence = tokenizer.decode(result)  

    # print('Input: {}'.format(inp))
    print('Predicted translation: {}'.format(predicted_sentence))
    print('Correct translation: {}'.format(tokenizer.decode(tf.squeeze(text, 0))))



val_data = tf.data.experimental.CsvDataset(filenames='cleaned-data/test-fixed.csv', 
                                record_defaults=[tf.string, tf.string],
                                header=True,
                                select_cols=[0, 1])

val_data = val_data.map(preprocessing_py_func)\
    .batch(1)\
    .prefetch(1)


for i, x in enumerate(val_data):
    if i > 9:
        break
    translate(*x)
    


