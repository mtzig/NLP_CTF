import torch
from torch import nn
from transformers import DistilBertForSequenceClassification


class BertClassifier(nn.Module):
    '''
    A simple wrapper around HuggingFace DistilBert model
    '''

    def __init__(self, num_labels=2, device='cpu'):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=num_labels).to(device)
        self.device = device

    def forward(self, X):
        X = X.to(self.device)
        input_id = X[:, 0]
        mask = X[:, 1]
        return self.bert(input_ids=input_id, attention_mask=mask)[0]


class CNNClassifier(nn.Module):
    '''
    Makes a simple 1D CNN text classifier based off Zhang et. al 2015 (BibTeX below)
    Expects a gensim KeyedVectors for embedding


    @article{zhang2015sensitivity,
        title={A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification},
        author={Zhang, Ye and Wallace, Byron},
        journal={arXiv preprint arXiv:1510.03820},
        year={2015}
    }
    '''

    def __init__(self, pretrained_embed, num_classes=2, num_feature_maps=100, kernel_sizes=(3, 4, 5), p_dropout=0.5, device='cpu'):
        super().__init__()

        embed_dim = pretrained_embed.shape[1]

        # self.embedder = nn.Embedding(num_embed, embed_dim, device=device)
        # self.embedder.requires_grad = False
        # self.embedder = nn.Embedding.from_pretrained(torch.from_numpy(KeyedVector.vectors).to(device))
        self.embedder = nn.Embedding.from_pretrained(pretrained_embed.to(device))


        self.convs = nn.ModuleList([nn.Conv1d(
            embed_dim, num_feature_maps, kernel, device=device) for kernel in kernel_sizes])

        self.dropout = nn.Dropout(p=p_dropout)

        self.classifier = nn.Linear(
            num_feature_maps * len(kernel_sizes), num_classes, device=device)

    def conv_pool(self, X, conv):

        # (B, F, L') where F is num_feature_maps, L' is length of convolved features
        X = conv(X)

        # (B, F, 1) -- global maxpool
        X = nn.functional.max_pool1d(X, X.shape[2])

        # (B, F) -- remove extra dim
        out = X.squeeze(2)

        return out

    def forward(self, X):
        '''
        INPUTS

        X: shape (B, L) where
                B is batch size
                L is length of sentences (should be 300)

        '''

        # (B, E, L) where E is embed dim i.e. 300
        embeds = self.embedder(X).permute(0, 2, 1)

        # (B, number of kernels * F)
        features = torch.cat([self.conv_pool(embeds, conv)
                             for conv in self.convs], dim=1)

        # non linearity
        features = nn.functional.relu(features)

        # dropout
        features = self.dropout(features)

        # fully connected layer
        logit = self.classifier(features)

        return logit
