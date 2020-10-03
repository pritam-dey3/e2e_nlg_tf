import tensorflow as tf
from transformers import AutoTokenizer
import re
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_slot_value_dict(mr):
    slot_value_pat = re.compile(r'(?:\ *)([a-zA-Z ]+)(\[[\w£\- ]+\],*)')
    return {'<' + m.group(1) + '>': re.sub(r'\]', '', m.group(2)[1:-1]) for m in re.finditer(slot_value_pat, mr)}


def preprocessing(mr, text):
    # mr, text = data
    mr = mr.numpy().decode('utf-8'); text = text.numpy().decode('utf-8')
    sv_dict = get_slot_value_dict(mr)
    keys = sv_dict.keys()
    sent = list(keys)
    slot_sent = ['<pad>'] * len(sent)
    if '<customer rating>' in keys:
        cr = tokenizer.tokenize(sv_dict['<customer rating>'])
        sent += cr
        slot_sent += ['<cr_slot>'] * len(cr)
    if '<priceRange>' in keys:
        pr = tokenizer.tokenize(sv_dict['<priceRange>'])
        sent += pr
        slot_sent += ['<pr_slot>'] * len(pr)
    if '<familyFriendly>' in keys and (sv_dict['<familyFriendly>'] != 'yes'):
        sent.remove('<familyFriendly>')
        sent.insert(0, '<notfamilyFriendly>')
    sent.insert(0, '<sos>')
    
    named_entities = ['<area>', '<eatType>', '<food>', '<near>', '<name>']
    label_text = '<sos>' + text
    for ne in named_entities:
        if ne not in keys:
            continue
        label_text = re.sub(sv_dict[ne], ne, label_text)
    sent = tokenizer.encode(sent, padding='max_length', max_length=32, return_tensors='tf')
    slot_sent = tokenizer.encode(slot_sent, padding='max_length', max_length=32, return_tensors='tf')
    label_text = tokenizer.encode(label_text, padding='max_length', max_length=100, return_tensors='tf')

    sent = tf.squeeze(sent, 0)
    slot_sent = tf.squeeze(slot_sent, 0)
    label_text = tf.squeeze(label_text, 0)
        
    return sent, slot_sent, label_text


def preprocessing_py_func(mr, text):
    sent, slot_sent, label_text = tf.py_function(preprocessing,
                                                inp=[mr, text],
                                                Tout=(tf.int32,) * 3)
    sent.set_shape([32])
    slot_sent.set_shape([32])
    label_text.set_shape([100])
    return sent, slot_sent, label_text   


train_data = tf.data.experimental.CsvDataset(filenames='cleaned-data/train-fixed.no-ol.csv', 
                                record_defaults=[tf.string, tf.string],
                                header=True,
                                select_cols=[0, 1])

tokenizer = AutoTokenizer.from_pretrained("t5-small")

special_tokens = {'additional_special_tokens': ['<area>', '<eatType>', '<food>', '<near>',                                                      '<name>', '<customer rating>', '<priceRange>',                                                   '<familyFriendly>', '<notfamilyFriendly>',                                                       '<cr_slot>', '<pr_slot>', '<sos>']}
tokenizer.add_special_tokens(special_tokens)

train_data = train_data.map(preprocessing_py_func).shuffle(buffer_size=15).batch(2).prefetch(1)

def show():
    for i, v in enumerate(train_data):
        print(tokenizer.decode(v[0][0, :]))
        print(tokenizer.decode(v[1][0, :]))
        print(tokenizer.decode(v[2][0, :]))
        if i == 1:
            break
