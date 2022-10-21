import argparse
import torch
import gc
import numpy as np
import pandas as pd
from process_data import get_jigsaw_datasets, init_embed_lookup, get_ctf_datasets, get_CivilComments_Datasets, get_jigsaw_dev_data
from models import CNNClassifier
from train_eval import train, evaluate, CTF
from loss import CLP_loss, ERM_loss

parser = argparse.ArgumentParser()
parser.add_argument('train_method', help='method to train model i.e. baseline, blind, augment, CLP')
parser.add_argument('--verbose', '-v', action='store_true', help='Print results')
parser.add_argument('--trials', '-t', default=10, help='The number of trials to run, defaults to 10')
parser.add_argument('--epochs', '-e', default=5, help='The number of epochs to train model, defaults to 5')
parser.add_argument('--test_name', '-n', default='test', help='The name of the test to run (defaults to "test"), the output files will be saved in the directory ./[name].csv')
parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() 
                                                     else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() 
                                                     else 'cpu', help='The device Pytorch should use cuda, mps or cpu')

# TODO(???): add arguments to adjust optimizer hyperparameters

args = parser.parse_args()


DEVICE = args.device

# load word2vec into gensim model
embed_lookup = init_embed_lookup()
pretrained_embed = torch.from_numpy(embed_lookup.vectors)

# get datasets
if args.train_method == 'CLP':
    train_data, A = get_jigsaw_datasets(device=DEVICE, data_type='CLP', embed_lookup=embed_lookup)
else:
    train_data = get_jigsaw_datasets(device=DEVICE, data_type=args.train_method, embed_lookup=embed_lookup)

jig_dev_data = get_jigsaw_dev_data(device=DEVICE, embed_lookup=embed_lookup)

# initialize civil commemts test dataset, so that we can evaluate performance on that too
cc_data = get_CivilComments_Datasets(device=DEVICE)

# initialize every ctf datasets
# TODO: implement code to get synth_toxic and synth_nontoxic
ctf_datas = []
for dataset in ('civil_eval', 'civil_train'):#, 'synth_toxic', 'synth_nontoxic'):
    ctf_datas.append(get_ctf_datasets(device=DEVICE, dataset=dataset, embed_lookup=embed_lookup))

# load into dataloader
train_loader =  torch.utils.data.DataLoader(train_data, batch_size=64)
jig_loader = torch.utils.data.DataLoader(jig_dev_data, batch_size=64)

cc_loader = torch.utils.data.DataLoader(cc_data, batch_size=64)

ctf_loaders = []
for data in ctf_datas:
    ctf_loaders.append(torch.utils.data.DataLoader(data, batch_size=64))

results = []

for trial in range(int(args.trials)):
    print(f'Trial {trial+1}/{int(args.trials)}')

    # first we do garbage collection,
    # as torch sometimes does not free model when we reinitialize it
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    # initialize models    
    model = CNNClassifier(pretrained_embed,device=DEVICE)
    if args.train_method == 'CLP':
        loss_fn = CLP_loss(torch.nn.CrossEntropyLoss(), A)
    else:
        loss_fn = ERM_loss(torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.AdamW(model.parameters())

    # train model
    for epoch in range(int(args.epochs)):
        print(f'Epoch {epoch+1}/{int(args.epochs)}')
        train(train_loader, model, loss_fn, optimizer, verbose=args.verbose)

    # evaluate loss/accuracy/sensitivity/specificity/AUC on Jigsaw dev set
    jig_results = evaluate(jig_loader, model, get_loss=True, verbose=args.verbose)

    # evaluate loss/accuracy/sensitivity/specificity/AUC on civil comments test set
    cc_results = evaluate(cc_loader, model, get_loss=True, verbose=args.verbose)

    # evaluate CTF gap over every eval dataset
    ctf_gaps = []
    for ctf_loader in ctf_loaders:
        ctf_gaps.append(CTF(ctf_loader, model))

    # TODO: evaluate tp, tn on training identity in Civil Comments

    results.append(jig_results+cc_results+tuple(ctf_gaps))


# output results as csv
columns = ('jig_loss', 'jig_accuracy', 'jig_tp', 'jig_tn', 'jig_auc',
            'cc_loss', 'cc_accuracy', 'cc_tp', 'cc_tn', 'cc_auc',
            'ctf_cc_eval', 'ctf_cc_train',
            #'ctf_synth_toxic', 'ctf_synth_nontoxic'
            )

df_results = pd.DataFrame(np.array(results), columns=columns)
df_results.to_csv(f'{args.test_name}.csv', index=False)