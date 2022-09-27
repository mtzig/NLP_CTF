import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import DistilBertTokenizer
from datasets import SimpleDataset


def load_jigsaw(file_path='../data/jigsaw', device='cpu', demo_mode=False):
    '''
    Loads the jigsaw (training set in the paper) into Pytorch datasets

    Input: 
    file_path: Location of train, test csv
    device: device to store labels on
    demo_mode: demonstration mode, only reads in a part of dataset so that traiing is faster

    Output:
    datasets: A tuple of size two
                    (train_dataset, test_dataset)
    '''
    # Create df with train data
    df_train = pd.read_csv(f'{file_path}/train.csv')

    # Create df with test data
    df_test = pd.read_csv(f'{file_path}/test.csv');
    df_test_labels = pd.read_csv(f'{file_path}/test_labels.csv')
    df_test['toxic'] = df_test_labels['toxic']
    df_test = df_test[df_test['toxic'] != -1]
    df_test.reset_index(inplace=True)

    if demo_mode:
        np.random.seed(3)
        df_train = df_train.sample(frac=1).reset_index()
        df_test = df_test.sample(frac=1).reset_index()


        df_train = pd.concat((df_train[df_train['toxic'] == 1].iloc[:128],
                            df_train[df_train['toxic'] == 0].iloc[:128]))

        df_test = pd.concat((df_test[df_test['toxic'] == 1].iloc[:128],
                            df_test[df_test['toxic'] == 0].iloc[:128]))


    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

    datasets = []
    for df in (df_train, df_test):

        # tokenization
        ids = []
        masks = []
        for txt in tqdm(df['comment_text'].values):
            token = tokenizer(txt, padding='max_length', max_length=300,
                               truncation=True, return_tensors="pt")
            ids.append(token['input_ids'])
            masks.append(token['attention_mask'])

        ids = torch.cat(ids)
        masks = torch.cat(masks)
        
        #features are not moved to device to save space
        features = torch.stack((ids, masks), dim=1)
        labels = torch.from_numpy(df['toxic'].values).to(device).long()

        datasets.append(SimpleDataset(features, labels))

    return datasets



        
