import pandas as pd
import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from datasets import SimpleDataset




def get_jigsaw_datasets(file_path='./data', device='cpu', data_type='baseline'):

    
    # Create df with train data
    df_train = pd.read_csv(f'{file_path}/train_df_blind.csv')


    if data_type == 'blind':
        df_train = None
    elif data_type == 'augment':
        df_train = None

    
    # Create df with dev data
    df_dev = pd.read_csv(f'{file_path}/jigsaw/test.csv')
        
    df_dev_labels = pd.read_csv(f'{file_path}/jigsaw/test_labels.csv')
    df_dev['toxic'] = df_dev_labels['toxic']
    df_dev = df_dev[df_dev['toxic'] != -1]
    df_dev.reset_index(inplace=True)


    embed_lookup = init_embed_lookup()
     
    datasets = []

    for df in (df_train, df_dev):

        padded_id = []
        for comment in df['comment_text']:
            seq = tokenize(comment)
            id = get_id(seq, embed_lookup)
            padded_id.append(pad_seq(id))

        X = torch.tensor(padded_id, device=device)
        y = torch.tensor(df['toxic'], device=device)

        dataset = SimpleDataset(X, y)

        datasets.append(dataset)

    return datasets


def init_embed_lookup(file_path='./data/GoogleNews-vectors-negative300.bin'):
    return KeyedVectors.load_word2vec_format(file_path, binary=True)



def get_id(seq, embed_lookup):

    seq_id = []
    for word in seq:
    
        try:
            idx = embed_lookup.key_to_index[word]
        except: 
            idx = 0
        seq_id.append(idx)

    return seq_id



def pad_seq(seq):
    if len(seq) > 300:
        return seq[:300]
    else:
        return seq + [0] * (300 - len(seq))



