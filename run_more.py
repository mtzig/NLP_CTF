import argparse
import torch
import gc
import numpy as np
import pandas as pd
from process_data import get_jigsaw_datasets, init_embed_lookup, get_ctf_datasets, get_CivilComments_Datasets, get_jigsaw_dev_data, get_CivilComments_idents_Datasets
from models import CNNClassifier
from train_eval import train, evaluate, CTF
from loss import CLP_loss, ERM_loss
from torch.utils.data import DataLoader


'''
This script runs same experiments, but records results at each epoch and averages
results of each trial

'''

parser = argparse.ArgumentParser()
parser.add_argument('train_method', help='method to train model i.e. baseline, blind, augment, CLP')
parser.add_argument('--lambda_clp', '-l', default=0.05, help='the lambda value, only applicable if CLP is used, defaults to 0.05')
parser.add_argument('--nontoxic', '-x', action='store_true', help='only uses nontoxic comments, only applicable if CLP is used')
parser.add_argument('--verbose', '-v', action='store_true', help='Print results')
parser.add_argument('--trials', '-t', default=3, help='The number of trials to run, defaults to 10')
parser.add_argument('--epochs', '-e', default=15, help='The number of epochs to train model, defaults to 5')
parser.add_argument('--test_name', '-n', default='test', help='The name of the test to run (defaults to "test"), the output files will be saved in the directory ./[name].csv')
parser.add_argument('--glove', '-g', action='store_false', help='uses glove instead of word2vec')
parser.add_argument('--featuremaps', '-f', default=100, help='number of feature maps for model to use, defaults to 100')
parser.add_argument('--kernelsizes', '-k', default=(2,3,4,5), nargs="+", type=int, help='kernel sizes to use to intialize model defaults to (2,3,4,5)')
parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() 
                                                     else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() 
                                                     else 'cpu', help='The device Pytorch should use cuda, mps or cpu')

# TODO(???): add arguments to adjust optimizer hyperparameters

args = parser.parse_args()

DEVICE = args.device

# load word2vec into gensim model
print('loading word embeddings')
embed_lookup = init_embed_lookup(word2vec=args.glove)
pretrained_embed = torch.from_numpy(embed_lookup.vectors)
print('done')
print('loading datasets')
# get datasets
if args.train_method == 'CLP':
    train_data, A = get_jigsaw_datasets(device=DEVICE, data_type='CLP', embed_lookup=embed_lookup)
else:
    train_data = get_jigsaw_datasets(device=DEVICE, data_type=args.train_method, embed_lookup=embed_lookup)

jig_dev_data = get_jigsaw_dev_data(device=DEVICE, embed_lookup=embed_lookup)

# initialize civil commemts test dataset, so that we can evaluate performance on that too
cc_data = get_CivilComments_Datasets(device=DEVICE, embed_lookup=embed_lookup)
cc_idents_data = get_CivilComments_idents_Datasets(device=DEVICE, embed_lookup=embed_lookup)

# initialize every ctf datasets
ctf_datas = []
for dataset in ('civil_eval', 'civil_train', 'synth_toxic', 'synth_nontoxic'):
    ctf_datas.append(get_ctf_datasets(device=DEVICE, dataset=dataset, embed_lookup=embed_lookup))

# load into dataloader
train_loader = DataLoader(train_data, batch_size=64)
jig_loader = DataLoader(jig_dev_data, batch_size=64)

cc_loader = DataLoader(cc_data, batch_size=64)
cc_idents_loader = DataLoader(cc_idents_data, batch_size=64)

ctf_loaders = []
for data in ctf_datas:
    ctf_loaders.append(DataLoader(data, batch_size=64))
print('done')

results = []

for trial in range(int(args.trials)):
    print('{:=^50}'.format(f'Trial {trial+1}/{int(args.trials)}'))

    trial_results = []

    print('initializing model')
    # first we do garbage collection,
    # as torch sometimes does not free model when we reinitialize it
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    # initialize models    
    model = CNNClassifier(pretrained_embed,device=DEVICE, num_feature_maps=args.featuremaps, kernel_sizes=args.kernelsizes)
    if args.train_method == 'CLP':
        loss_fn = CLP_loss(torch.nn.CrossEntropyLoss(), A, lmbda=float(args.lambda_clp), only_nontox=args.nontoxic)
    else:
        loss_fn = ERM_loss(torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.AdamW(model.parameters())

    print('done')
    # train model
    for epoch in range(int(args.epochs)):
        print(f'Epoch {epoch+1}/{int(args.epochs)}')
        train(train_loader, model, loss_fn, optimizer, verbose=args.verbose)

        print('evaluating model')
        # evaluate loss/accuracy/sensitivity/specificity/AUC on Jigsaw dev set
        jig_results = evaluate(jig_loader, model, get_loss=True, verbose=args.verbose)

        # evaluate loss/accuracy/sensitivity/specificity/AUC on civil comments test set
        cc_results = evaluate(cc_loader, model, get_loss=True, verbose=args.verbose)

        # evaluate loss/accuracy/sensitivity/specificity/AUC on civil comments idents only test set
        cc_idents_results = evaluate(cc_idents_loader, model, get_loss=True, verbose=args.verbose)

        # evaluate CTF gap over every eval dataset
        ctf_gaps = []
        for ctf_loader in ctf_loaders:
            ctf_gaps.append(CTF(ctf_loader, model,))

        trial_results.append(jig_results+cc_results+cc_idents_results+tuple(ctf_gaps))

    results.append(trial_results)



# output results as csv
columns = ('jig_loss', 'jig_accuracy', 'jig_tp', 'jig_tn', 'jig_auc',
            'cc_loss', 'cc_accuracy', 'cc_tp', 'cc_tn', 'cc_auc',
            'cci_loss', 'cci_accuracy', 'cci_tp', 'cci_tn', 'cci_auc',
            'ctf_cc_eval', 'ctf_cc_train',
            'ctf_synth_toxic', 'ctf_synth_nontoxic',
            )

print('outputting results to csv')

# averages results over trials
results = np.average(np.array(tuple(zip(*results))), axis=0)

df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv(f'{args.test_name}.csv', index=False)
print('done')
print('experiment finished')