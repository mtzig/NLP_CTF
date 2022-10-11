import pandas as pd
import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from datasets import SimpleDataset
import tqdm




def get_jigsaw_datasets(file_path='./data', device='cpu', data_type='baseline'):

    
    # Create df with train data
    df_train = pd.read_csv(f'{file_path}/jigsaw/train_with_idents.csv')


    if data_type == 'blind':
        df_train = process_blind(df_train)
    elif data_type == 'augment':
        df_train = process_augment(df_train)

    
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



def process_blind(df):
    '''
    Preprocess dataframe to have identity tokens masked as identity
    '''


    idents = ['gay', 'bisexual', 'transgender', 'trans',
       'queer', 'lgbt', 'lgbtq', 'homosexual', 'straight', 'heterosexual',
       'male', 'female', 'nonbinary', 'african', 'african american', 'black',
       'white', 'european', 'hispanic', 'latino', 'latina', 'latinx',
       'mexican', 'canadian', 'american', 'asian', 'indian', 'middle eastern',
       'chinese', 'japanese', 'christian', 'muslim', 'jewish', 'buddhist',
       'catholic', 'protestant', 'sikh', 'taoist', 'old', 'older', 'young',
       'younger', 'teenage', 'millenial', 'middle aged', 'elderly', 'blind',
       'deaf', 'paralyzed', 'lesbian']

        # Adding identity column to train_df_short (either works I think)
    df['identity'] = (df[idents].sum(axis=1) > 0).astype(int)
    

        # Creating train_df_blind by dropping all identiy columns from train_df_short

    # Replacing identities in comment text with an identity token
    token = "identity"
    for row_index in tqdm(range(len(df))):
        if df.at[row_index, "identity"] == 1:
            for identity in idents:
                if df.at[row_index, identity] == 1 :
                    #TODO: Handle cases such as 'transparent'
                    df.at[row_index, "comment_text"] = df.at[row_index, "comment_text"].replace(identity, token)
    return df


def process_augment(df):

    idents = ['gay', 'bisexual', 'transgender', 'trans',
       'queer', 'lgbt', 'lgbtq', 'homosexual', 'straight', 'heterosexual',
       'male', 'female', 'nonbinary', 'african', 'african american', 'black',
       'white', 'european', 'hispanic', 'latino', 'latina', 'latinx',
       'mexican', 'canadian', 'american', 'asian', 'indian', 'middle eastern',
       'chinese', 'japanese', 'christian', 'muslim', 'jewish', 'buddhist',
       'catholic', 'protestant', 'sikh', 'taoist', 'old', 'older', 'young',
       'younger', 'teenage', 'millenial', 'middle aged', 'elderly', 'blind',
       'deaf', 'paralyzed', 'lesbian']

    # Adding identity column to train_df_short (either works I think)
    df['identity'] = (df[idents].sum(axis=1) > 0).astype(int)


    df_identities = df[df.identity==1].reset_index()
    comment_list = []
    toxic_list = []
    augmented = []

    for row_index in tqdm(range(len(df_identities))):
        for identity in idents:
            if df_identities.at[row_index, identity] == 1:
                comment_list.append(df_identities.at[row_index, "comment_text"])
                toxic_list.append(df_identities.at[row_index, "toxic"])
                augmented.append(0)
                for diff_identity in idents:
                    if diff_identity == identity:
                        break
                    comment_list.append(df_identities.at[row_index, "comment_text"].replace(identity, diff_identity))
                    toxic_list.append(df_identities.at[row_index, "toxic"])
                    augmented.append(1)
                    
    data_tuples = list(zip(comment_list, toxic_list))
    train_df_augment = pd.DataFrame(data_tuples, columns=['comment_text','toxic'])

    df_nonidents = df[df.identity==0].reset_index()['comment_text', 'toxic']
    # train_df_augment['augmented'] = augmented
    return df.concat((train_df_augment, df_nonidents), ignore_index=True)


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



