import csv
import os.path

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np

plt.switch_backend('agg')


class CsvLogger:
    def __init__(self, filepath='./', filename='results.csv', data=None):
        self.log_path = filepath
        self.log_name = filename
        self.csv_path = os.path.join(self.log_path, self.log_name)
        self.fieldsnames = ['epoch', 'val_error1', 'val_error5', 'val_loss', 'train_error1', 'train_error5',
                            'train_loss']

        with open(self.csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writeheader()

        self.data = {}
        for field in self.fieldsnames:
            self.data[field] = []
        if data is not None:
            for d in data:
                d_num = {}
                for key in d:
                    d_num[key] = float(d[key]) if key != 'epoch' else int(d[key])
                self.write(d_num)

    def write(self, data):
        for k in self.data:
            self.data[k].append(data[k])
        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writerow(data)

    def save_params(self, args, params):
        with open(os.path.join(self.log_path, 'params.txt'), 'w') as f:
            f.write('{}\n'.format(' '.join(args)))
            f.write('{}\n'.format(params))

    def write_text(self, text, print_t=True):
        with open(os.path.join(self.log_path, 'params.txt'), 'a') as f:
            f.write('{}\n'.format(text))
        if print_t:
            print(text)

    def plot_progress_errk(self, claimed_acc=None, title='MobileNetv2', k=1):
        tr_str = 'train_error{}'.format(k)
        val_str = 'val_error{}'.format(k)
        plt.figure(figsize=(9, 8), dpi=300)
        plt.plot(self.data[tr_str], label='Training error')
        plt.plot(self.data[val_str], label='Validation error')
        if claimed_acc is not None:
            plt.plot((0, len(self.data[tr_str])), (1 - claimed_acc, 1 - claimed_acc), 'k--',
                     label='Claimed validation error ({:.2f}%)'.format(100. * (1 - claimed_acc)))
        plt.plot((0, len(self.data[tr_str])),
                 (np.min(self.data[val_str]), np.min(self.data[val_str])), 'r--',
                 label='Best validation error ({:.2f}%)'.format(100. * np.min(self.data[val_str])))
        plt.title('Top-{} error for {}'.format(k, title))
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.xlim(0, len(self.data[tr_str]) + 1)
        plt.savefig(os.path.join(self.log_path, 'top{}.png'.format(k)))

    def plot_progress_loss(self, title='MobileNetv2'):
        plt.figure(figsize=(9, 8), dpi=300)
        plt.plot(self.data['train_loss'], label='Training')
        plt.plot(self.data['val_loss'], label='Validation')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.xlim(0, len(self.data['train_loss']) + 1)
        plt.savefig(os.path.join(self.log_path, 'loss.png'))

    def plot_progress(self, claimed_acc1=None, claimed_acc5=None, title='MobileNetv2'):
        self.plot_progress_errk(claimed_acc1, title, 1)
        self.plot_progress_errk(claimed_acc5, title, 5)
        self.plot_progress_loss(title)
        plt.close('all')
