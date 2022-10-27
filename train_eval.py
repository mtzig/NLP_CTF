
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from process_data import init_embed_lookup, get_id, pad_seq
from gensim.utils import tokenize


def train(dataloader, model, loss_fn, optimizer, verbose=False):
    '''
    Train the model for one epoch
    dataloader: The dataloader for the training data
    model: The model to train
    loss_fn: The loss function to use for training
    optimizer: The optimizer to use for training
    verbose: Whether to print the average training loss of the epoch
    :param scheduler: Learning rate scheduler to use
    '''
    model.train()

    data_iter = iter(dataloader)

    avg_loss = 0

    num_batches = len(dataloader)

    for minibatch in tqdm(data_iter):


        loss = loss_fn(model, minibatch)
        avg_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()


        optimizer.step()


    avg_loss /= num_batches

    if verbose:
        print("Average training loss:", avg_loss)

def evaluate(dataloader, model, get_loss=False, verbose=False):
    '''
    Evaluate the model's accuracy, sensitivities and specificity (assumes binary classification), AUC

    Input:
    dataloader: The dataloader for the validation/testing data
    model: The model to evaluate

    get_loss: Calculates the average cross-entropy loss
    verbose: Whether to print the results

    Output:
    A tuple containing the overall accuracy and the sensitivity/specificity, AUC

    '''

    model.eval()
    data_iter = iter(dataloader)

    sensitivity = 0
    specificity = 0
    accuracy = 0

    positive_count = 0

    num_batches = len(dataloader)


    true_y = []
    pred_y = []

    with torch.no_grad():
        

        if get_loss:
            loss = 0
            loss_fn = torch.nn.CrossEntropyLoss()

        for minibatch in tqdm(data_iter):
           
            X, y = minibatch[:2]

            pred = model(X)

            true_y.append(y)
            pred_y.append(pred[:,1]) #only care about toxic pred
            
            # PyTorch does not support tensor indexing on metal, so need to move to cpu
            if pred.device.type == 'mps':
                pred = pred.to('cpu')
                y = y.to('cpu')

            positive_count += (y==1.0).sum().item()

            sensitivity += (pred.argmax(1)[y==1.0] == y[y==1.0]).type(torch.float).sum().item()
            specificity += (pred.argmax(1)[y==0.0] == y[y==0.0]).type(torch.float).sum().item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

            if get_loss:
                # accumulate loss over entire epoch
                loss += loss_fn(pred, y)

        if get_loss:
            loss /= num_batches

        sensitivity /= positive_count
        specificity /= (len(dataloader.dataset) - positive_count)
        accuracy /= len(dataloader.dataset)

        true_y = torch.cat(true_y).cpu().numpy()
        pred_y = torch.cat(pred_y).cpu().numpy()

        auc = roc_auc_score(true_y, pred_y)


    if verbose:
        if get_loss:
            print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}, AUC: {auc}')

    if get_loss:
        return loss.item(), accuracy, sensitivity, specificity, auc

    return accuracy, sensitivity, specificity, auc

def CTF(dataloader, model, verbose=False):
    '''
        Calculates the CTF gap

        model: the model to evaluate CTF on
        dataloader: the data to evaluate CTF on
    '''
    
    model.eval()

    data_iter = iter(dataloader)
    
    cum_gap = 0
    num_examples = 0

    with torch.no_grad():

        for (X,A) in tqdm(data_iter):

            # this is redundant to do every iteration, but whatever
            l, i ,w = A.shape
            
            A_preds = torch.nn.functional.softmax(model(A.reshape(-1, w)), 1).reshape(l, i, -1)[:,:,0]
            X_preds = torch.unsqueeze(torch.nn.functional.softmax(model(X), 1)[:,0], 1)

            cum_gap += torch.sum(torch.abs(X_preds - A_preds))

            num_examples += l * i

    ctf_gap =  (cum_gap / num_examples).item()

    if verbose:
        print(f'CTF gap: {ctf_gap}')

    return ctf_gap

def get_pred(comment_text, model, embed_lookup=None):
    '''
    On input string

    returns its logit and probability of being toxic
    '''

    if not embed_lookup:
        embed_lookup = init_embed_lookup()

    DEVICE = next(model.parameters()).device

    seq = tokenize(comment_text)
    id = pad_seq(get_id(seq, embed_lookup))
    
    input = torch.tensor(id, device=DEVICE).unsqueeze(0)

    model.eval()

    with torch.no_grad():

        logit = model(input)
        pred = torch.nn.functional.softmax(logit, dim=1)[0,1]

    return logit.tolist()[0], pred.item()

