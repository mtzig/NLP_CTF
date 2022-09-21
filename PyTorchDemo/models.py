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

