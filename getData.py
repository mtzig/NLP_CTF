import pandas as pd
import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from datasets import SimpleDataset




def get_jigsaw_datasets(file_path='./data', device='cpu', data_set = 'None'):

    
    # Create df with train data
    df_train = pd.read_csv(f'{file_path}/jigsaw/train.csv')
    # Create df with test data
    df_test = pd.read_csv(f'{file_path}/jigsaw/test.csv')


    if data_set == 'blind':
        df_train = pd.read_csv(f'{file_path}/train_df_blind.csv')
    elif data_set == 'augment':
        df_train = pd.read_csv(f'{file_path}/train_df_synthetic.csv')

    df_test_labels = pd.read_csv(f'{file_path}/jigsaw/test_labels.csv')
    df_test['toxic'] = df_test_labels['toxic']
    df_test = df_test[df_test['toxic'] != -1]
    df_test.reset_index(inplace=True)

    # ints = []
    embed_lookup = init_embed_lookup()
     
    datasets = []

    for df in (df_train, df_test):

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



