from torch import nn
from transformers import DistilBertForSequenceClassification

class BertClassifier(nn.Module):
    '''
    A simple wrapper around HuggingFace DistilBert model
    '''

    def __init__(self, num_labels=2, device='cpu'):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).to(device)
        self.device = device

    def forward(self, X):
        X = X.to(self.device)
        input_id = X[:,0] 
        mask = X[:,1]
        return self.bert(input_ids= input_id, attention_mask=mask)[0]


class ConvPool1D(nn.Module):
    '''
    Initializes a layer of conv and pool
    '''

    def __init__(self, ,device='cpu'):
        super().__init__()

        

    def forward(self, X):




class CNNClassifier(nn.Module):
    '''
    Makes a simple 1D CNN text classifier
    Expects a gensim KeyedVectors for embedding
    '''

    def __init__(self, KeyedVector, device='cpu') -> None:
        super().__init__()
        
        num_embed, embed_dim = KeyedVector.vectors.shape

        self.embeder =  nn.Embedding(num_embed, embed_dim, device=device)
        self.featurizer = nn.Conv1d(_, 5, device=device)
        self.classifier = 


    def forward(self, X):
        '''
        '''
