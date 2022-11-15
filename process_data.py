import pandas as pd
import torch
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from torch.utils.data import TensorDataset
from tqdm import tqdm
import re
import numpy as np


idents = list(pd.read_csv('./data/random_split_data/train_identities.txt', header=None).iloc[:,0].astype('string'))


def get_CivilComments_Datasets(device='cpu', embed_lookup=None):
    '''
    gets the test split of civil comments dataset
    '''

    if not embed_lookup:
        embed_lookup = init_embed_lookup()

    CC_df = pd.read_csv('./data/civil_comments/civil_comments.csv', index_col=0)
    CC_df['toxicity'] = (CC_df['toxicity'] >= 0.5).astype(int)


    sub_df = CC_df[CC_df['split'] == 'test']


    padded_id = []
    for comment in tqdm(sub_df['comment_text'].values):
        seq = tokenize(comment)
        id = get_id(seq, embed_lookup)
        padded_id.append(pad_seq(id))
    
    features = torch.tensor(padded_id, device=device)

    labels = torch.from_numpy(sub_df['toxicity'].values).to(device)
    labels = labels.to(device).long()
    
    return TensorDataset(features, labels)

def get_Synthetic_Datasets(device='cpu', embed_lookup=None, synth_df_name="89"):
    '''
    gets the test split of civil comments dataset
    '''

    if not embed_lookup:
        embed_lookup = init_embed_lookup()
        
    if synth_df_name == "89":
        df_path = './data/bias_madlibs_89k.csv'
    else:
        df_path = './data/bias_madlibs_77k.csv'

    CC_df = pd.read_csv(df_path, index_col=0)
    CC_df['toxicity'] = (CC_df['Label'] == "BAD").astype(int)


    sub_df = CC_df


    padded_id = []
    for comment in tqdm(sub_df['Text'].values):
        seq = tokenize(comment)
        id = get_id(seq, embed_lookup)
        padded_id.append(pad_seq(id))
    
    features = torch.tensor(padded_id, device=device)

    labels = torch.from_numpy(sub_df['toxicity'].values).to(device)
    labels = labels.to(device).long()
    
    return TensorDataset(features, labels)

def get_CivilComments_idents_Datasets(device='cpu', embed_lookup=None):
    '''
        returns subset of CivilComments dataset only with identity
    '''

    if not embed_lookup:
        embed_lookup = init_embed_lookup()

    df_nontoxic = pd.read_csv(f'./data/civil_comments/civil_train_data.csv', index_col=0)
    df_toxic = pd.read_csv(f'./data/civil_comments/civil_toxic_train_data.csv', index_col=0)

    df_nontoxic['toxicity'] = 0
    df_toxic['toxicity'] = 1

    df_idents = pd.concat([df_nontoxic[['comment_text', 'toxicity']], df_toxic[['comment_text', 'toxicity']]])

    padded_id = []
    for comment in tqdm(df_idents['comment_text'].values):
        seq = tokenize(comment)
        id = get_id(seq, embed_lookup)
        padded_id.append(pad_seq(id))
    
    features = torch.tensor(padded_id, device=device)

    labels = torch.from_numpy(df_idents['toxicity'].values).to(device)
    labels = labels.to(device).long()
    
    return TensorDataset(features, labels)

def get_Synthetic_idents_Datasets(device='cpu', embed_lookup=None):
    '''
        returns subset of CivilComments dataset only with identity
    '''

    if not embed_lookup:
        embed_lookup = init_embed_lookup()

    df_nontoxic = pd.read_csv(f'./data/synthetic/synthetic_nontoxic_df_2.csv', index_col=0)
    df_toxic = pd.read_csv(f'./data/synthetic/synthetic_toxic_df_2.csv', index_col=0)

    df_nontoxic['Label'] = 0
    df_toxic['Label'] = 1

    df_idents = pd.concat([df_nontoxic[['Text', 'Label']], df_toxic[['Text', 'Label']]])

    padded_id = []
    for comment in tqdm(df_idents['Text'].values):
        seq = tokenize(comment)
        id = get_id(seq, embed_lookup)
        padded_id.append(pad_seq(id))
    
    features = torch.tensor(padded_id, device=device)

    labels = torch.from_numpy(df_idents['Label'].values).to(device)
    labels = labels.to(device).long()
    
    return TensorDataset(features, labels)



def get_jigsaw_dev_data(file_path='./data', device='cpu', embed_lookup=None):
    '''
    returns the dev split of jigsaw dataset
    '''

    if not embed_lookup:
        embed_lookup = init_embed_lookup()
    
    # Create df with dev data
    df_dev = pd.read_csv(f'{file_path}/jigsaw/test.csv')
        
    df_dev_labels = pd.read_csv(f'{file_path}/jigsaw/test_labels.csv')
    df_dev['toxic'] = df_dev_labels['toxic']
    df_dev = df_dev[df_dev['toxic'] != -1]
    df_dev.reset_index(inplace=True)

    padded_id = []
    for comment in tqdm(df_dev['comment_text']):
        seq = tokenize(comment)
        id = get_id(seq, embed_lookup)
        padded_id.append(pad_seq(id))

    X = torch.tensor(padded_id, device=device)
    y = torch.tensor(df_dev['toxic'], device=device)

    return TensorDataset(X, y)

def get_jigsaw_datasets(file_path='./data', device='cpu', data_type='baseline', embed_lookup=None):    
    '''
    return datasets of the form X,y,M where M is metadata

    M is only meaningful when data_type is baseline

    baseline: returns baseline data
    blind: returns data with blind preprocessing
    augment: retuns data with augment preprocessing
    CLP: returns A, a tensor of adversarially perturbed examples
    
    '''
    if not embed_lookup:
        embed_lookup = init_embed_lookup()

    # Create df with train data
    df_train = pd.read_csv(f'{file_path}/jigsaw/train.csv')
    df_train = df_train.drop(df_train.columns[3:8], axis=1)

    # add identity to column indicating precence of identity in sentence
    if data_type != 'baseline':
        for row_index, row in enumerate(df_train.itertuples()):
            for identity in idents:
                regex = re.compile(r'\b' + re.escape(identity) + r'\b')
                if regex.search(row[2]):
                    df_train.at[row_index, identity] = 1
                else:
                    df_train.at[row_index, identity] = 0
    
    if data_type == 'blind':
        df_train = process_blind(df_train)
    elif data_type == 'augment':
        df_train = process_augment(df_train)
    elif data_type == 'CLP': # CLP
        df_train, df_adversarial = process_clp(df_train)
   
    #only need metadata for CLP, otherwise we just use some dummy data
    if data_type == 'CLP':
        M = torch.tensor(df_train['index'], device=device)
    else:
        M = torch.zeros(len(df_train), device=device) 

    padded_id = []
    for comment in tqdm(df_train['comment_text']):
        seq = tokenize(comment)
        id = get_id(seq, embed_lookup)
        padded_id.append(pad_seq(id))

    X = torch.tensor(padded_id, device=device)
    y = torch.tensor(df_train['toxic'], device=device)

    dataset = TensorDataset(X, y, M)

    # need to get the adversarial matrix
    if data_type == 'CLP':        
        tokenized_adversarials = []

        # tokenize every sentence in A
        for row in tqdm(df_adversarial.itertuples(), total=len(df_adversarial)):
            row_adv = []
            for comment in row[3:]:
                seq = tokenize(comment)
                id = get_id(seq, embed_lookup)
                row_adv.append(pad_seq(id))
            tokenized_adversarials.append(row_adv)

        A = torch.tensor(tokenized_adversarials, device=device)
        return dataset, A

    return dataset

def get_ctf_datasets(file_path='./data', dataset='civil_eval', device='cpu', embed_lookup=None):
    '''
        returns datasets to be used for CTF metric

        civil_test: civil comment  non-toxic held out identities
        civil_train: civil comment non-toxic training identities

        TODO implement getting different datasets
    '''
    
    if not embed_lookup:
        embed_lookup = init_embed_lookup()

    if dataset == 'civil_eval':
        df = pd.read_csv(f'{file_path}/civil_comments/civil_test_data.csv', index_col=0)
    elif dataset == 'civil_train':
        df = pd.read_csv(f'{file_path}/civil_comments/civil_train_data.csv', index_col=0)

    # TODO: implement this synthetic -- do the processing on the fly using synthetic

    
    elif dataset == 'synth_nontoxic':
        df = pd.read_csv(f'{file_path}/synthetic/synthetic_nontoxic_df.csv', index_col=0)
    elif dataset == 'synth_nontoxic_2':
        df = pd.read_csv(f'{file_path}/synthetic/synthetic_nontoxic_df_2.csv', index_col=0)
    elif dataset == 'synth_toxic_2':
        df = pd.read_csv(f'{file_path}/synthetic/synthetic_toxic_df_2.csv', index_col=0)
    else:
        df = pd.read_csv(f'{file_path}/synthetic/synthetic_toxic_df.csv', index_col=0)


    X_comments = []
    A_comments = []

    # assume every column except comment_text is for an identity
    num_idents = len(df.columns) - 1
    num_sents = len(df)


    for row in tqdm(df.itertuples(), total=len(df)):

        # first tokenize/ get ID of comment
        comment = row[1]
        seq = tokenize(comment)
        id = get_id(seq, embed_lookup)
        X_comments.append(pad_seq(id))

        # next do the same with adversarially perturbed sentences
        perturbed_sentences = row[2:]
        for perturbed in perturbed_sentences:
            seq = tokenize(perturbed)
            id = get_id(seq, embed_lookup)
            A_comments.append(pad_seq(id))
            

    X = torch.tensor(X_comments, device=device)
    A = torch.tensor(A_comments, device=device).reshape(num_sents, num_idents, -1)
    
    dataset = TensorDataset(X, A)

    return dataset

def process_blind(df):
    '''
    Preprocess dataframe to have identity tokens masked as identity
    '''

    # Adding identity column to train_df_short (either works I think)
    df['identity'] = (df[idents].sum(axis=1) > 0).astype(int)

    # Replacing identities in comment text with an identity token
    token = "identity"
    for row_index in tqdm(range(len(df))):

        if df.at[row_index, "identity"] == 1:
            for identity in idents:
                if df.at[row_index, identity] == 1 :
                    regex = r'\b' + re.escape(identity) + r'\b'
                    df.at[row_index, "comment_text"] = re.sub(regex, token, df.at[row_index, "comment_text"], flags=re.IGNORECASE)
    return df


def process_augment(df):

    # Adding identity column to train_df_short (either works I think)
    df['identity'] = (df[idents].sum(axis=1) > 0).astype(int)

    df_identities = df[df.identity==1].reset_index()
    comment_list = []
    toxic_list = []

    for row_index in tqdm(range(len(df_identities))):
        for identity in idents:
            regex = r'\b' + re.escape(identity) + r'\b'
            if df_identities.at[row_index, identity] == 1:
                comment_list.append(df_identities.at[row_index, "comment_text"])
                toxic_list.append(df_identities.at[row_index, "toxic"])
                for diff_identity in idents:
                    if diff_identity == identity:
                        continue
                    comment_list.append(re.sub(regex, diff_identity, df_identities.at[row_index, "comment_text"], flags=re.IGNORECASE))
                    toxic_list.append(df_identities.at[row_index, "toxic"])
     
    data_tuples = list(zip(comment_list, toxic_list))
    train_df_augment = pd.DataFrame(data_tuples, columns=['comment_text','toxic'])

    df_nonidents = df[df.identity==0][['comment_text', 'toxic']].reset_index()
    # train_df_augment['augmented'] = augmented
    return pd.concat((train_df_augment, df_nonidents), ignore_index=True)

def process_clp(df):
    
    df['identity'] = (df[idents].sum(axis=1) > 0).astype(int)

    df_idents = df[df.identity==1].reset_index(drop=True)
    df_idents['index'] = np.arange(len(df_idents))

    df_nonidents = df[df.identity==0].reset_index(drop=True)
    df_nonidents['index'] = -1
    
    identity_regex = re.compile('|'.join(idents), re.IGNORECASE)
    
    a = []
    for comment_text in tqdm(df_idents['comment_text']):
        identity = identity_regex.search(comment_text)[0].lower()

        # generate adversarial
        cur_a = []
        for diff_identity in idents:
            if diff_identity == identity:
                continue
            cur_a.append(comment_text.replace(identity, diff_identity))
        a.append(cur_a)

    df_adversarial = pd.DataFrame(list(zip(*a))).T

    return pd.concat((df_idents, df_nonidents), ignore_index=True), df_adversarial
    


def init_embed_lookup(word2vec=True, file_path='./data'):
    '''
    intializes the embeddings

    either word2vec or glove
    '''
    if word2vec:
        return KeyedVectors.load_word2vec_format(f'{file_path}/GoogleNews-vectors-negative300.bin', binary=True)
    
    return KeyedVectors.load_word2vec_format(f'{file_path}/glove_840B_300d.txt', binary=False, no_header=True)


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



