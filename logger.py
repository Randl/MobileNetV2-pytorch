import csv
import os.path

from matplotlib import pyplot as plt
import numpy as np


class CsvLogger:
    def __init__(self, filepath='./', filename='results.csv'):
        self.log_path = filepath
        self.log_name = filename
        self.csv_path = os.path.join(self.log_path, self.log_name)
        self.fieldsnames = ['epoch', 'val_error1', 'val_error5', 'val_loss', 'train_error1', 'train_error5',
                            'train_loss']
        self.data = {}
        for field in self.fieldsnames:
            self.data[field] = []
        with open(self.csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writeheader()

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

    def plot_progress_err(self, claimed_acc, title='MobileNetv2'):
        plt.figure(figsize=(18, 16), dpi=120)
        plt.plot(self.data['train_error1'], label='Training')
        plt.plot(self.data['val_error1'], label='Validation')
        plt.plot((0, len(self.data['train_error1'])), (1 - claimed_acc, 1 - claimed_acc), 'k--',
                 label='Claimed accuracy')
        plt.plot((0, len(self.data['train_error1'])),
                 (np.min(self.data['val_error1']), np.min(self.data['val_error1'])), 'r--',
                 label='Top validation error')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.xlim(0, len(self.data['train_error1']) + 1)
        plt.savefig(os.path.join(self.log_path, 'acc.png'))

    def plot_progress_loss(self, title='MobileNetv2'):
        plt.figure(figsize=(18, 16), dpi=120)
        plt.plot(self.data['train_loss'], label='Training')
        plt.plot(self.data['val_loss'], label='Validation')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.xlim(0, len(self.data['train_loss']) + 1)
        plt.savefig(os.path.join(self.log_path, 'loss.png'))
