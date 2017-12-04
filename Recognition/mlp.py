#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from scipy.io import loadmat
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import pdb
from preprocessing import load_collect
from matplotlib import pyplot as plt
import seaborn as sns

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        fcs = []
        for n_in, n_out in zip(layer_sizes[:-1],
                               layer_sizes[1:]):
            fcs.append(nn.Linear(n_in, n_out))
        self.fcs = nn.ModuleList(fcs)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = F.relu(fc(x))
            if i < len(self.fcs) - 1:
                x = F.relu(x)
        return F.log_softmax(x)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        if not data.shape[0] == target.shape[0]:
            raise ValueError('data and target should have the same shape')

        self._data = data
        self._target = target
        self._N = data.shape[0]

    def __len__(self):
        return self._N

    def __getitem__(self, item):
        return (torch.from_numpy(self._data[item]), int(self._target[item]))


def plot_classification(y_pred, y_test, n_class):
    results = np.zeros([n_class, n_class])
    for i in range(len(y_test)):
        results[y_test[i]][y_pred[i]] += 1
    results = results / np.sum(results, 1, keepdims=True)
    print('the classification results', results)
    sns.heatmap(results, annot=True, cmap="YlGnBu")
    plt.show(block=True)
    return results


if __name__ == '__main__':
    with open(__file__, 'r') as f:
        print(f.read())

    parser = argparse.ArgumentParser(description='HARU')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='disable CUDA use')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='coefficient for weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epoch number for training')
    parser.add_argument('--log_interval', type=float, default=10,
                        help='interval batches for logging')
    parser.add_argument('--f_interval', type=float, default=40,
                        help='the interval to select features')
    parser.add_argument('--n_feature', type=float, default=40,
                        help='feature lengths')
    args = parser.parse_args()

    # set seed
    cuda_seed = 1234
    np_seed = 1234
    np.random.seed(np_seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        print('using cuda')
        torch.cuda.manual_seed(cuda_seed)

    data, label = load_collect(args.n_feature, args.f_interval)
    data = data.astype('float32')

    indices = np.random.permutation(data.shape[0])
    train_num = int(data.shape[0] * 0.9)
    train_loader = torch.utils.data.DataLoader(
        Dataset(data[indices[:train_num]], label[indices[:train_num]]),
        batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        Dataset(data[indices[train_num:]], label[indices[train_num:]]),
        batch_size=100, shuffle=False)

    N, nx = data.shape
    n_class = 5
    layer_sizes = [nx, 20, n_class]

    # model
    model = MLP(layer_sizes)
    if use_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    def adjust_learning_rate(optimizer):
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(epoch):
        if epoch == int(args.epochs / 2):
            adjust_learning_rate(optimizer)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model.forward(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]
                ))

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        y_preds, y_tests = [], []
        for _, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model.forward(data)
            test_loss = test_loss + F.nll_loss(output, target, size_average=False).data[0]
            pred = output.data.max(1)[1]
            correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

            y_preds.append(pred.numpy().reshape(-1))
            y_tests.append(target.data.numpy().reshape(-1))

        test_loss = test_loss / len(test_loader.dataset)
        print('\n Test set: Average loss: {:4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        ))

        if epoch == args.epochs:
            y_preds = np.concatenate(y_preds, axis=-1)
            y_tests = np.concatenate(y_tests, axis=-1)
            plot_classification(y_preds, y_tests, n_class)

    # def my_test():
    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     for data, target in my_test_loader:
    #         if use_cuda:
    #             data, target = data.cuda(), target.cuda()
    #         data, target = Variable(data), Variable(target)
    #         output = model.forward(data)
    #         test_loss = test_loss + F.nll_loss(output, target, size_average=False).data[0]
    #         pred = output.data.max(1, keepdim=True)[1]
    #         print(pred.numpy().reshape(-1))
    #         correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()
    #
    #     test_loss = test_loss / len(my_test_loader.dataset)
    #     print('\n Test set: Average loss: {:4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(my_test_loader.dataset),
    #         100. * correct / len(my_test_loader.dataset)
    #     ))

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # my_test()
