import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def mx_to_torch_sparse_tensor(mx):
    """Convert a np matrix to a torch sparse tensor."""
    idxs = np.where(mx>0)
    indices = torch.from_numpy(
        np.vstack((idxs[0], idxs[1])).astype(np.int64))
    values = torch.from_numpy(mx)
    shape = torch.Size(mx.shape)
    import pdb
    pdb.set_trace()
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    '''
    Input:
        1. output = model output logits
        2. labels = model
    Output: accuracy
    Description:
        Gets the argmax output and compares matches them to labels.
        Accuracy = number of correct classes over number of samples
    Purpose:
        Calculate accuracy during training
    '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val):
    '''
    Input:
        1. epoch = epoch id
        2. model = pytorch model to be trained
        3. optimizer = optimizer corresponding to the model and loss
        4. features = features to be used by the model
        5. adj = adjacency matrix to be used by the model
        6. labels = labels that correspond to the features 
        7. idx_train = indices for training loss
        8. idx_val = indices for validation during training
    Output: model
        Return the model object after training
    Description:
        1. Features -> model -> softmax output -> negative log loss (only on train idxs) -> backpropagation
        2 .Evaluate val idxs
    Purpose:
        Run one epoch of training and evaluate on a val set for loss and accuracy
    '''
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return model
    
def test(model, test_dict):
    '''
    Input:
        1. model = pytorch model to be tested
        2. test_dict = dictionary of inputs to be used for testing (consistent with train_dict)
    Output: None
    Description:
        1. Features -> model -> softmax output -> negative log loss (only on test idxs)
        2. Evaluate on test idxs
    Purpose:
        Takes a set of testing inputs and get loss and accuracy
    '''
    features = test_dict['features']
    adj = test_dict['adj']
    labels = test_dict['labels']
    idx_test = test_dict['idx_test']
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return
    
def run_train(model, train_dict):
    '''
    Input:
        1. model = pytorch model to be trained
        2. train_dict = dictionary of inputs to be used for training
    Output: model
        Return the model object after training
    Description:
        Takes a set of training inputs and run training over a specified number of epochs
    Purpose:
        Run training over multiple epochs using given parameters
    '''
    # Train model
    optimizer = train_dict['optimizer']
    features = train_dict['features']
    adj = train_dict['adj']
    labels = train_dict['labels']
    idx_train = train_dict['idx_train']
    idx_val = train_dict['idx_val']
    t_total = time.time()
    for epoch in range(train_dict['num_epochs']):
        model = train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    return model