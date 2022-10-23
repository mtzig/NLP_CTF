import pandas as pd
import torch
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from torch.utils.data import TensorDataset
from tqdm import tqdm
import re

idents = ['gay', 'bisexual', 'transgender', 'trans',
       'queer', 'lgbt', 'lgbtq', 'homosexual', 'straight', 'heterosexual',
       'male', 'female', 'nonbinary', 'african', 'black',
       'white', 'european', 'hispanic', 'latino',
       'buddhist', 'catholic', 'protestant', 'sikh', 'taoist', 
       'old', 'older', 'young',
       'younger', 'teenage', 'millenial', 'elderly', 'blind',
       'deaf', 'paralyzed', 'lesbian']


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
    df_train = pd.read_csv(f'{file_path}/jigsaw/train_with_idents.csv')

    if data_type == 'blind':
        df_train = process_blind(df_train)
    elif data_type == 'augment':
        df_train = process_augment(df_train)

   
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
        df_adversarial = pd.read_csv(f'{file_path}/jigsaw/train_adversarials.csv')
        
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
        
def process_synthetic(toxic):
        synthetic_data = pd.read_csv("./data/train_df_synthetic.csv")
        test_identities_df = pd.read_csv("./data/test_identities.txt", header=None)
        test_identities_list = test_identities_df.values.tolist()

        test_identities = []
        for ident in test_identities_list:
            test_identities.append(ident[0])

        sentences = []
        a = []
        toxicity = []

        for row_index in tqdm(range(len(synthetic_data))):
            comment_text = synthetic_data.iloc[row_index]['comment_text'].split()
            if toxic:
                if synthetic_data.iloc[row_index]['toxic'] >= 0.5 and len(set(test_identities).intersection(comment_text)) != 0:
                    identity = str(set(test_identities).intersection(comment_text).pop())
                    sentences.append(synthetic_data.iloc[row_index]['comment_text'])
                    toxicity.append(0)
                    cur_a = []
                    
                    for diff_identity in test_identities:
                        cur_a.append(synthetic_data.at[row_index, "comment_text"].replace(identity, diff_identity))
                    a.append(cur_a)
            else:
                if synthetic_data.iloc[row_index]['toxic'] < 0.5 and len(set(test_identities).intersection(comment_text)) != 0:
                    identity = str(set(test_identities).intersection(comment_text).pop())
                    sentences.append(synthetic_data.iloc[row_index]['comment_text'])
                    toxicity.append(0)
                    cur_a = []
                    
                    for diff_identity in test_identities:
                        cur_a.append(synthetic_data.at[row_index, "comment_text"].replace(identity, diff_identity))
                    a.append(cur_a)

        synthetic_raw = pd.DataFrame(list(zip(*a)))
        synthetic_df = synthetic_raw.T
        synthetic_df.insert(0, column='comment_text', value=sentences)

        return synthetic_df

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

    df_nonidents = df[df.identity==0][['comment_text', 'toxic']].reset_index()
    # train_df_augment['augmented'] = augmented
    return pd.concat((train_df_augment, df_nonidents), ignore_index=True)



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



