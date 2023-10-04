# Script Credit: https://github.dev/dayu11/Availability-Attacks-Create-Shortcuts/tree/main
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from constants import *
from datasets import construct_train_dataset, PoisonWithCleanDataset

def normalize_01(tsr):
    maxv = torch.max(tsr)
    minv = torch.min(tsr)
    return (tsr-minv)/(maxv-minv)

def train(train_data, train_targets, net, optimizer):
    loss_func = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    inputs, targets = train_data, train_targets
    def closure():
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        outputs = net(inputs)
        loss = loss_func(outputs, targets)

    train_loss = loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = predicted.eq(targets.data).float().cpu().sum()
    acc = 100.*float(correct)/float(total)
    return (train_loss, acc)

def main():
    parser = argparse.ArgumentParser(description='Fit perturbations with simple models')
    parser.add_argument('poison_identifier', type=str, 
                        help='Poison identifier (from DATA_SETUPS in constants.py). Determines poison perturbations to test for linear separability.')
    parser.add_argument('--hidden_layers', default=0, type=int, 
                        help='number of hidden layers')
    args = parser.parse_args()
    print('args:', args)
    print('setup_key:', args.poison_identifier)
    baseset = construct_train_dataset(setup_key='cifar10', 
                                      normalize=False,
                                      transforms_key='test_transform')

    if args.poison_identifier == 'clean':
        labels = torch.tensor(baseset.targets)
        perturbations = torch.tensor(baseset.data / 255.0, dtype=torch.float32)
    else:
        pd = construct_train_dataset(setup_key=args.poison_identifier, 
                                     normalize=False,
                                     transforms_key='test_transform')
        ds = PoisonWithCleanDataset(baseset, pd)
        perturbed_x = []
        clean_x = []
        labels = []
        for i in tqdm(range(len(ds)), desc='collecting perturbations'):
            perturbed_x.append(ds[i][0])
            clean_x.append(ds[i][2])
            labels.append(torch.tensor(ds[i][1], dtype=torch.long))
        perturbed_x = torch.stack(perturbed_x)
        clean_x = torch.stack(clean_x)
        labels = torch.stack(labels)
        assert perturbed_x.shape == clean_x.shape, 'perturbed_x and clean_x should have the same shape but are {} and {}'.format(perturbed_x.shape, clean_x.shape)
        perturbations = perturbed_x - clean_x
    
    print('perturbations shape:', perturbations.shape)
    print('labels shape:', labels.shape)
    perturbations = perturbations.cuda()
    labels = labels.cuda()

    assert perturbations.shape[0] == labels.shape[0], 'number of perturbations and labels do not match ({} vs {})'.format(perturbations.shape[0], labels.shape[0])
    assert perturbations.dtype == torch.float32, 'perturbations are not float32'
    assert labels.dtype == torch.long, 'labels are not long'

    
    train_data = normalize_01(perturbations)
    train_targets = labels

    num_classes = 10 # CIFAR-10 dataset

    module_list = [nn.Flatten()]
    input_dim = np.prod(train_data.shape[1:])

    hidden_width = 30
    for i in range(args.hidden_layers):
        module_list.append(nn.Linear(input_dim, hidden_width))
        module_list.append(nn.Tanh())
        input_dim = hidden_width

    module_list += [nn.Linear(input_dim, num_classes)]

    net = nn.Sequential(*module_list)
    net = net.cuda()
    optimizer = optim.LBFGS(net.parameters(), lr=0.5) 

    train_accs = []
    train_losses = []
    for step in range(500):
        train_loss, train_acc = train(train_data, train_targets, net, optimizer)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        # print(f'step {step}: train_loss: {train_loss:.3f}, train_acc: {train_acc:.2f}')
    print('final training loss: %.3f'%train_loss, 'final training acc: %.2f'%train_acc)

    # save plot of logistic regression training accuracy vs number of steps
    plt.plot(train_accs)
    plt.xlabel('step')
    plt.ylabel('training accuracy')
    plt.title(f'Final Train Accuracy ({args.poison_identifier}): {train_acc:.2f}%')
    plt.savefig(f'test-linear-sep-{args.poison_identifier}.png')

if __name__ == '__main__':
    main()