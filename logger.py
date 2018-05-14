import csv
import os.path

class CsvLogger:
    def __init__(self, filepath='./', filename='results.csv'):
        self.log_path = filepath
        self.log_name = filename
        self.csv_path = os.path.join(self.log_path, self.log_name)
        self.fieldsnames = ['epoch', 'val_error1',  'val_loss', 'train_error1', 'train_loss'] #'val_error5',
        with open(self.csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writeheader()

    def write(self, data):
        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writerow(data)

    def save_params(self, args, params):
        with open(os.path.join(self.log_path, 'params.txt'), 'w') as f:
            f.write('{}\n'.format(' '.join(args)))
            f.write('{}\n'.format(params))