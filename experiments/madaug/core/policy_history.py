"""PolicyHistory, copied verbatim from official MADAug utils.py (commit 279b60a).

[madaug-adapt] Only this class is copied: official utils.py also imports seaborn at module
level, which MDAAug does not need. MDAAug.__init__ instantiates PolicyHistory; the official
training script never calls save()/plot().
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PolicyHistory(object):

    def __init__(self, op_names, save_dir, n_class):
        self.op_names = op_names
        self.save_dir = save_dir
        self._initialize(n_class)

    def _initialize(self, n_class):
        self.history = []
        # [{m:[], w:[]}, {}]
        for i in range(n_class):
            self.history.append({'magnitudes': [],
                                'weights': [],
                                'var_magnitudes': [],
                                'var_weights': []})

    def add(self, class_idx, m_mu, w_mu, m_std, w_std):
        if not isinstance(m_mu, list):  # ugly way to bypass batch with single element
            return
        self.history[class_idx]['magnitudes'].append(m_mu)
        self.history[class_idx]['weights'].append(w_mu)
        self.history[class_idx]['var_magnitudes'].append(m_std)
        self.history[class_idx]['var_weights'].append(w_std)

    def save(self, class2label=None):
        path = os.path.join(self.save_dir, 'policy')
        vis_path = os.path.join(self.save_dir, 'vis_policy')
        os.makedirs(path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        header = ','.join(self.op_names)
        for i, history in enumerate(self.history):
            k = i if class2label is None else class2label[i]
            np.savetxt(f'{path}/policy{i}({k})_magnitude.csv',
                       history['magnitudes'], delimiter=',', header=header, comments='')
            np.savetxt(f'{path}/policy{i}({k})_weights.csv',
                       history['weights'], delimiter=',', header=header, comments='')
            np.savetxt(f'{vis_path}/policy{i}({k})_var_magnitude.csv',
                       history['var_magnitudes'], delimiter=',', header=header, comments='')
            np.savetxt(f'{vis_path}/policy{i}({k})_var_weights.csv',
                       history['var_weights'], delimiter=',', header=header, comments='')

    def plot(self):
        PATH = self.save_dir
        mag_file_list = glob.glob(f'{PATH}/policy/*_magnitude.csv')
        weights_file_list = glob.glob(f'{PATH}/policy/*_weights.csv')
        n_class = len(mag_file_list)

        f, axes = plt.subplots(n_class, 2, figsize=(15, 5*n_class))

        for i, file in enumerate(mag_file_list):
            df = pd.read_csv(file).dropna()
            x = range(0, len(df))
            y = df.to_numpy().T
            axes[i][0].stackplot(x, y, labels=df.columns, edgecolor='none')
            axes[i][0].set_title(file.split('/')[-1][:-4])

        for i, file in enumerate(weights_file_list):
            df = pd.read_csv(file).dropna()
            x = range(0, len(df))
            y = df.to_numpy().T
            axes[i][1].stackplot(x, y, labels=df.columns, edgecolor='none')
            axes[i][1].set_title(file.split('/')[-1][:-4])

        axes[-1][-1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2), fancybox=True, shadow=True, ncol=10)
        plt.savefig(f'{PATH}/policy/schedule.png')

        f, axes = plt.subplots(1, 1, figsize=(7,5))

        frames = []
        for i, file in enumerate(mag_file_list):
            df = pd.read_csv(file).dropna()
            df['class'] = file.split('/')[-1][:-4].split('_')[0]
            frames.append(df.tail(1))

        df = pd.concat(frames)
        df.set_index('class').plot(ax=axes, kind='bar', stacked=True, legend=False, rot=90, fontsize=8)
        axes.set_ylabel("magnitude")
        plt.savefig(f'{PATH}/policy/magnitude_by_class.png')
        
        f, axes = plt.subplots(1, 1, figsize=(7,5))
        frames = []
        for i, file in enumerate(weights_file_list):
            df = pd.read_csv(file).dropna()
            df['class'] = file.split('/')[-1][:-4].split('_')[0].split('(')[1][:-1]
            frames.append(df.tail(1))

        df = pd.concat(frames)   
        df.set_index('class').plot(ax=axes, kind='bar', stacked=True, legend=False, rot=90, fontsize=8)
        axes.set_ylabel("probability")
        axes.set_xlabel("")
        plt.savefig(f'{PATH}/policy/probability_by_class.png')
        
        return f
