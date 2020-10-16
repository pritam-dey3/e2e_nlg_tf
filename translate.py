from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from data_preprocessing import get_slot_value_dict
import re
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("t5-small")

special_tokens = {'additional_special_tokens': ['<area>', '<eatType>', '<food>', '<near>',                                                      '<name>', '<customer rating>', '<priceRange>',                                                   '<familyFriendly>', '<notfamilyFriendly>',                                                       '<cr_slot>', '<pr_slot>', '<sos>']}
tokenizer.add_special_tokens(special_tokens)

end_token = tokenizer.convert_tokens_to_ids('</s>')

data = np.load("output.npy")
data = data[1:,:]

df = pd.read_csv("cleaned-data/test-fixed.csv", header=0, usecols=[0,1])

def preprocessing(mr):
    # mr, text = data
    # mr = mr.numpy().decode('utf-8'); text = text.numpy().decode('utf-8')
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

    sent = tokenizer.encode(sent, padding='max_length', max_length=32, return_tensors='tf')
    slot_sent = tokenizer.encode(slot_sent, padding='max_length', max_length=32, return_tensors='tf')
    return sent, slot_sent, sv_dict

lines = np.array(["mr", "pred", "gold"])

for i in tqdm(range(data.shape[0])):
    mr, text = df.iloc[i,:]
    sent, slot_sent, sv_dict = preprocessing(mr)
    tokens = data[i, :]
    endidx = np.where(tokens == end_token)
    if endidx[0].shape == (0,): endidx = (np.array([100]), )

    pred = tokenizer.decode(tokens[:endidx[0][0]])

    for k in sv_dict.keys():
        pred = re.sub(k, sv_dict[k], pred)
    pred = re.sub('<sos>', '', pred)
    newline = np.array([sv_dict, pred, text])

    lines = np.vstack((lines, newline))


np.save("prediction.npy", lines)