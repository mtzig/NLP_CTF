
from tqdm import tqdm
import torch

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
    Evaluate the model's accuracy, sensitivities and specificity (assumes binary classification)

    Input:
    dataloader: The dataloader for the validation/testing data
    model: The model to evaluate

    get_loss: Calculate the average cross-entropy loss as well as the accuracy and subclass sensitivities
    verbose: Whether to print the results

    Output:
    A tuple containing the overall accuracy and the sensitivity/specificity

    '''

    model.eval()
    data_iter = iter(dataloader)

    sensitivity = 0
    specificity = 0
    accuracy = 0

    positive_count = 0

    num_batches = len(dataloader)


    with torch.no_grad():
        

        if get_loss:
            loss = 0
            loss_fn = torch.nn.CrossEntropyLoss()

        for (X,y) in tqdm(data_iter):
           

            pred = model(X)
            
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

    if verbose:
        if get_loss:
            print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}')

    if get_loss:
        return loss, accuracy, sensitivity, specificity

    return accuracy, sensitivity, specificity

def CTF(dataloader, model):
    
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

    return cum_gap / num_examples


