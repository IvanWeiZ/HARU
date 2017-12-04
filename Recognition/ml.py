#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.utils.data
import argparse
import numpy as np
from preprocessing import load_collect
from load_cloud import load_cloud
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sns


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
    # name = ['downstair', 'still', 'upstair', 'walking', 'running']
    print('the classification results', results)
    sns.heatmap(results, annot=True, cmap="YlGnBu")
    plt.show(block=True)
    return results

def run():
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
    parser.add_argument('--f_interval', type=float, default=50,
                        help='the interval to select features')
    parser.add_argument('--n_feature', type=float, default=50,
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
    label = label.astype('int32')

    indices = np.random.permutation(data.shape[0])
    train_num = int(data.shape[0] * 0.9)
    # x_train, y_train = data[indices[:train_num]], label[indices[:train_num]]
    # x_test, y_test = data[indices[train_num:]], label[indices[train_num:]]
    x_train, y_train = data, label

    x_cloud, y_cloud, _ = load_cloud(args.n_feature, args.f_interval, '0')
    x_cloud = x_cloud.astype('float32')
    y_cloud = y_cloud.astype('int32')
    x_train, y_train = x_cloud, y_cloud
    # x_train, y_train = np.vstack([x_cloud, x_train]), np.hstack([y_cloud, y_train])

    x_cloud, y_cloud, time_cloud = load_cloud(args.n_feature, args.f_interval, '1')
    x_cloud = x_cloud.astype('float32')
    y_cloud = y_cloud.astype('int32')
    x_test, y_test = x_cloud, y_cloud

    print('x_train', x_train.shape)
    print('x_test', x_test.shape)

    N, nx = data.shape
    n_class = 5

    # #######################################################
    # # Logistic regression
    # #######################################################
    # model = LogisticRegression(penalty='l2', C=1./args.weight_decay)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test).astype('int32')
    # # acc = np.mean(y_pred == y_test)
    # acc = model.score(x_test, y_test)
    # print('logistic regression', 'acc', '{:.2f}%'.format(100*acc))
    #
    # # acc = model.score(x_cloud, y_cloud)
    # # print('logistic regression', 'acc', '{:.2f}%'.format(100 * acc))
    # plot_classification(y_pred, y_test, n_class)

    #######################################################
    # Knn
    #######################################################
    model = KNeighborsClassifier(n_neighbors=n_class)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).astype('int32')
    # acc = np.mean(y_pred == y_test)
    acc = model.score(x_test, y_test)
    print('Knn                ', 'acc', '{:.2f}%'.format(100*acc))
    return y_pred
    # acc = model.score(x_cloud, y_cloud)
    # print('Knn                ', 'acc', '{:.2f}%'.format(100 * acc))
    # plot_classification(y_pred, y_test, n_class)

    # #######################################################
    # # Decision Tree
    # #######################################################
    # model = DecisionTreeClassifier()
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test).astype('int32')
    # # acc = np.mean(y_pred == y_test)
    # acc = model.score(x_test, y_test)
    # print('Decision Tree      ', 'acc', '{:.2f}%'.format(100*acc))
    #
    # # acc = model.score(x_cloud, y_cloud)
    # # print('Decision Tree      ', 'acc', '{:.2f}%'.format(100 * acc))
    # plot_classification(y_pred, y_test, n_class)

import boto3
import table
import datetime

action_dict = {'going_down_stairs':0, 'staying_still':1, 'walking':2, 'going_up_stairs':3, 'running':4}
resverse_action_dict = {0:'going_down_stairs', 1:'staying_still', 2:'walking', 3:'going_up_stairs', 4:'running'}

def align():
    device = '1'
    dynamodb = boto3.resource('dynamodb')
    lines = table.scan_table_allpages("Location", "Devicename", device)
    lines = sorted(lines, key=lambda k: k['Timestamp'])

    def _convert_from_strf(time):
        return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S').timestamp()
    def _convert_to_strf(time):
        return datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')

    _, _, real_time = load_cloud(50, 50, '1')
    y_pred = run()
    ind1 = 0
    for i in range(len(lines)):
        if lines[i]['Timestamp'] < real_time[ind1]:
            lines[i]['PredAction'] = resverse_action_dict[y_pred[ind1]]
        else:
            ind1 = ind1 + 1
    a = 0

    # data = [{"Timestamp": l['Timestamp'], 'Action': l['Action'],
    #          'X': float(l['X']), 'Y': float(l['Y']), 'Z': float(l['Z']), 'Actiontime': l['Actiontime']}
    #         for l in lines]
    #
    # data = sorted(data, key=lambda k: k['Timestamp'])
    # dx = np.reshape([d['X'] for d in data], [-1, 1])
    # dy = np.reshape([d['Y'] for d in data], [-1, 1])
    # dz = np.reshape([d['Z'] for d in data], [-1, 1])
    # daction = [d['Action'] for d in data]
    # dtime = [d['Actiontime'] for d in data]


if __name__ == '__main__':
    align()