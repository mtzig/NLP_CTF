import argparse
import torch
from process_data import get_jigsaw_datasets, init_embed_lookup, get_ctf_datasets, get_CivilComments_Datasets, get_jigsaw_dev_data
from models import CNNClassifier
from train_eval import train, evaluate, CTF
from loss import CLP_loss, ERM_loss

parser = argparse.ArgumentParser()
parser.add_argument('train_method', help='method to train model i.e. baseline, blind, augment, CLP')
parser.add_argument('--verbose', '-v', action='store_true', help='Print results')
parser.add_argument('--trials', '-t', default=10, help='The number of trials to run, defaults to 10')
parser.add_argument('--epochs', '-e', default=5, help='The number of epochs to train model, defaults to 5')
parser.add_argument('--test_name', '-n', default='test', help='The name of the test to run (defaults to "test"), the output files will be saved in the directory ./[name]/')
parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() 
                                                     else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() 
                                                     else 'cpu', help='The device Pytorch should use cuda, mps or cpu')

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

# TODO(???): also initialize civil commemts test dataset, so that we can evaluate performance on that to do

# TODO: initialize every possible ctf datasets i.e. CC_eval, CC_train, synth_toxic, synth_nontoxic
ctf_data = get_ctf_datasets(device=DEVICE, embed_lookup=embed_lookup)

# load into dataloader
train_loader =  torch.utils.data.DataLoader(train_data, batch_size=64)
jig_loader = torch.utils.data.DataLoader(jig_dev_data, batch_size=64)

# TODO: load all possible ctf dataset
ctf_loader = torch.utils.data.DataLoader(ctf_data, batch_size=64)

for trial in range(int(args.trials)):
    print(f'Trial {trial+1}/{int(args.trials)}')

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

    # evaluate CTF gap
    # TODO: evaluate over all possible ctf datasets
    # TODO: add verbose flad to CTF
    ctf_gap = CTF(ctf_loader, model)

    # TODO: read values into a csv and save it to directory args.test_name


