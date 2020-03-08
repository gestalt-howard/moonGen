# Author: Aaron Wu / Howard Tai

# Script defining functions necessary for model training and evaluation

import os
import time
import torch

import numpy as np
import torch.optim as optim
import torch.nn.functional as F


def softmax(x):
    """
    Performs softmax over elements of vector x

    Input(s):
    - x (numpy ndarray)

    Output(s):
    - (numpy ndarray)
    """
    return np.exp(x) / np.transpose(np.tile(np.sum(np.exp(x), axis=1), (x.shape[1], 1)), [1, 0])


def accuracy(output, labels):
    """
    Input:
        1. output (PyTorch tensor) = model output logits
        2. labels (PyTorch tensor) = model

    Output: accuracy (float)

    Description:
        Gets the argmax output and compares matches them to labels.
        Accuracy = number of correct classes over number of samples

    Purpose:
        Calculate accuracy during training
    """
    preds = output.max(1)[1].type_as(labels)  # Second argument (index 1) are argmax indexes
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val):
    """
    Input:
        1. epoch (int) = epoch id
        2. model (PyTorch model object) = pytorch model to be trained
        3. optimizer (PyTorch optimizer object) = optimizer corresponding to the model and loss
        4. features (PyTorch tensor) = features to be used by the model, shape [N, M]
        5. adj (PyTorch tensor) = adjacency matrix to be used by the model, shape [N, N]
        6. labels (PyTorch tensor) = labels that correspond to the features, shape [N, 1]
        7. idx_train (PyTorch tensor) = indices for training loss, shape [num_train]
        8. idx_val (PyTorch tensor) = indices for validation during training, shape [num_val]

    Output: model
        Return the model object after training

    Description:
        1. Features -> model -> softmax output -> cross entropy (only on train idxs) -> backpropagation
        2. Run evaluation on val idxs

    Purpose:
        Run one epoch of training and evaluate on a val set for loss and accuracy
    """
    t = time.time()  # Start timer

    # Training mode
    model.train()
    optimizer.zero_grad()  # Initialize gradients to zero
    output = model(features, adj)  # Forward pass

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # nll == cross entropy
    acc_train = accuracy(output[idx_train], labels[idx_train])

    # Back-propagate
    loss_train.backward()
    optimizer.step()

    # Evaluation mode
    model.eval()
    output = model(features, adj)  # Forward pass

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # Status update
    print(
        'Epoch: {:04d}'.format(epoch + 1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),
        'time: {:.4f}s'.format(time.time() - t)
    )

    return model


def test(model, test_dict):
    """
    Input:
        1. model (PyTorch model object) = pytorch model to be tested
        2. test_dict (dict) = dictionary of inputs to be used for testing (consistent with train_dict)

    Output: None

    Description:
        1. Features -> model -> softmax output -> cross entropy (only on test idxs)
        2. Evaluate on test idxs

    Purpose:
        Takes a set of testing inputs and get loss and accuracy
    """
    # Unpack items
    features = test_dict['features']
    adj = test_dict['adj']
    labels = test_dict['labels']
    idx_test = test_dict['idx_test']  # PyTorch Tensor

    # Evaluation mode
    model.eval()
    output = model(features, adj)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # Status update
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return None


def run_train(model, train_dict):
    """
    Input:
        1. model (PyTorch model object) = pytorch model to be trained
        2. train_dict (dict) = dictionary of inputs to be used for training

    Output: model
        Return the model object after training

    Description:
        Takes a set of training inputs and run training over a specified number of epochs

    Purpose:
        Run training over multiple epochs using given parameters
    """
    # Unpack items
    optimizer = train_dict['optimizer']
    features = train_dict['features']
    adj = train_dict['adj']
    labels = train_dict['labels']
    idx_train = train_dict['idx_train']
    idx_val = train_dict['idx_val']

    t_total = time.time()  # Start timer

    # Run epochs
    for epoch in range(train_dict['num_epochs']):
        model = train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    return model
