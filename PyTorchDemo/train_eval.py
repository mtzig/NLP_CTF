
from tqdm import tqdm
import torch

def train(dataloader, model, loss_fn, optimizer, verbose=False, use_tqdm=False):
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

    steps_per_epoch = dataloader.batches_per_epoch()

    avg_loss = 0

    step_iter = tqdm(range(steps_per_epoch)) if use_tqdm else range(steps_per_epoch)

    for _ in step_iter:

        X, y = next(dataloader)

        loss = loss_fn(model(X), y)
        avg_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()


        optimizer.step()


    avg_loss /= steps_per_epoch

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

    # num_samples = np.zeros(num_subclasses)
    # subgroup_correct = np.zeros(num_subclasses)
    with torch.no_grad():
        steps_per_epoch = dataloader.batches_per_epoch()
        sensitivity = 0
        specificity = 0
        accuracy = 0

        positive_count = 0

        if get_loss:
            loss = 0
            loss_fn = torch.nn.CrossEntropyLoss()

        for i in tqdm(range(steps_per_epoch)):
           
            X, y = next(dataloader)

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
            loss /= steps_per_epoch

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