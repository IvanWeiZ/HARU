# pre-processing.py
from scipy.signal import butter, lfilter
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# D = 151
# Intervel = 100


def load_collect(D, Intervel):
    data = []
    labels = []
    for label, name in enumerate(['downstair', 'still', 'upstair', 'walking', 'running']):
        with open('collect/'+name, 'rb') as f:
            d = pickle.load(f)

        d = filter_array(d, D, Intervel)
        data.append(d)
        labels.append(np.ones(d.shape[0]) * label)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels


def filter_array(pp_data, D, Intervel):
    out = []
    for i in range(int(np.floor((pp_data.shape[0]-D) / float(Intervel)))):
        part = pp_data[Intervel * i: Intervel * i + D]
        fx, fy, fz = filtering(part[:, 0], part[:, 1], part[:, 2])
        # fx, fy, fz = part[:, 0], part[:, 1], part[:, 2]
        out.append(np.concatenate([fx, fy, fz], axis=0))
    return np.array(out)


def read(filename):
    with open(filename, 'r') as f:
        string = f.read()
    data = []
    index = 0
    while index < len(string) - 10:
        x_ind = string[index:].find('x')
        y_ind = string[index:].find('y')
        z_ind = string[index:].find('z')
        timestamp_ind = string[index:].find('timestamp')
        x = float(string[index + x_ind + 2: index + y_ind - 1])
        y = float(string[index + y_ind + 2: index + z_ind - 1])
        z = float(string[index + z_ind + 2: index + timestamp_ind - 1])
        next_x_ind = string[index + x_ind + 2:].find('x')
        if next_x_ind == -1:
            break
        # timestamp = float(string[index + timestamp_ind + 10: index + x_ind + next_x_ind - 2])
        data.append((10*x, 10*y, 10*z, None))
        index = index + x_ind + next_x_ind - 2
    return data
#
# def readmyown(filename):
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#     data = []
#     for l in lines:
#         if l[0] != 'X':
#             continue
#         x_ind = l.find('X')
#         y_ind = l.find('Y')
#         z_ind = l.find('Z')
#         x = float(l[x_ind + 4 : y_ind - 1])
#         y = float(l[y_ind + 4 : z_ind - 2])
#         z = float(l[z_ind + 4 : ])
#         data.append((x, y, z, None))
#     return data


def filtering(fx, fy, fz):
    assert fx.ndim==1 and fy.ndim==1 and fz.ndim==1 , 'fx fy fz must be 1-d'
    fc = 0.3 # filter cutoff
    fs = 50  # frequency rate of the signal
    [but, att] = butter(6, fc / (fs / 2.)) # Butterworth filter creation
    gx = lfilter(but, att, fx)
    gy = lfilter(but, att, fy)
    gz = lfilter(but, att, fz)
    fx = fx - gx
    fy = fy - gy
    fz = fz - gz
    return fx, fy, fz


def plot_acc():
    mlp = [92.86,  92.98,  92.37,  94.60,  94.86,   95.11,   94.67,  92.64,  94.48,  89.66,  87.31,  74.90,  62.89,  53.65]
    lr =  [62.86,  59.09,  57.25,  74.92,  78.00,   74.89,   75.05,  71.83,  70.98,  70.88,  65.40,  62.65,  62.47,  56.67]
    knn = [90.48,  92.15,  91.98,  94.57,  94.92,   96.00,   94.67,  96.83,  96.86,  96.64,  96.10,  93.88,  91.35,  80.91]
    dt =  [76.19,  83.88,  81.30,  86.86,  87.94,   87.11,   87.24,  87.44,  88.96,  89.02,  90.33,  87.55,  86.18,  82.45]
    n_features = [150, 130, 120, 100,  90, 70, 60, 40, 30, 20, 10, 5, 3, 1]
    n_features_log = [np.log(a) for a in n_features]

    sns.set()
    plt.plot(n_features, mlp, 'r')
    plt.plot(n_features, lr, 'g')
    plt.plot(n_features, knn, 'b')
    plt.plot(n_features, dt, 'k')
    plt.legend(['MLP', 'LR', 'KNN', 'DT'], loc=4)
    plt.show(block=True)


if __name__ == '__main__':
    plot_acc()
    exit(0)
    data = read('collect/running.txt')
    pp_data = np.array([[x, y, z] for x, y, z, _ in data])
    with open('collect/running', 'wb') as f:
        pickle.dump(pp_data, f)
    exit(0)