import torch

class CLP_loss:

    def __init__(self, loss_fn, A):

        self.loss_fn = loss_fn

        self.A = A

        # number of identity terms per comment is the 2nd dim of A
        self.num_idents = A.shape[1]
        self.pad_length = A.shape[2]

        self.device = A.device

    def __call__(self, model, minibatch):

        X, y, M = minibatch

        loss = self.loss_fn(model(X), y)
        
        ident_comments = (M != -1)
        num_ident_comments = sum(ident_comments)

        if num_ident_comments != 0:
            # we want to randomly sample a perturbed example for each comment
            sampler = torch.broadcast_to(torch.randint(self.num_idents, 
                                                    (num_ident_comments, 1, 1), device=self.device), 
                                        (num_ident_comments, 1, self.pad_length))

            # print(sampler.shape)

            # print(self.A[M[ident_comments]].gather(1, sampler).shape)

            loss += self.CLP(X[ident_comments], self.A[M[ident_comments]].gather(1, sampler).squeeze(1), model)

        return loss


    def CLP(self, X, A, model):
        '''
        X: l x ...
        A: l x ...
        '''

        # model.eval()

        A_out = model(A)
        X_out = model(X)

        loss = torch.sum(torch.abs(X_out - A_out))

        # model.train()


        return loss 

class ERM_loss:
    '''
    simple wrapper around torch loss so as to conform to our API
    '''


    def __init__(self, loss_fn):

        self.loss_fn = loss_fn

    def __call__(self, model, minibatch):

        X, y, _ = minibatch
        return self.loss_fn(model(X), y)