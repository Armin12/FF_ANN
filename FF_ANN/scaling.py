
import numpy as np

class Standardization:
    def __init__(self):
        self.mean_data = 0
        self.std_data = 1

    def fit(self, train_data):
        self.mean_data = np.mean(train_data, axis=0)
        self.std_data = np.std(train_data, axis=0)
        return self.mean_data, self.std_data

    def transform(self, data):
        return (data - self.mean_data) / self.std_data
