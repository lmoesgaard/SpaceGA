import os
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, List, Dict


@dataclass
class DataManager:
    files: Dict
    file_names: List
    n: int


def filter_incomplete(data):
    for name in list(data.keys()):
        if data[name]["x"] is None or data[name]["y"] is None:
            del data[name]
    return data


class DataGen:
    def __init__(self, train_dir, val_dir, device, batch_size=2048):
        self.batch_size = batch_size
        self.device = device

        self.data = {dtype: self.find_inputs(dir) for dtype, dir in
                     zip(["val", "train"], [val_dir, train_dir])}

        self.i = 0
        self.mode = "val"
        self.X_data, self.y_data = self._load_data()

    def find_inputs(self, directory):
        data = {}
        for file in os.listdir(directory):
            details = file.split("_")
            name = details[0]
            if name not in data:
                data[name] = {"x": None, "y": None}
            x_or_y = details[1][0]
            data[name][x_or_y] = os.path.join(directory, file)
        data = filter_incomplete(data)
        return DataManager(files=data, file_names=list(data.keys()), n=len(data.keys()))

    def _load_data(self):
        files = self._get_current_files()
        x, y = (self._load_array(files, x_or_y) for x_or_y in ["x", "y"])
        n = len(x)
        shuffle = np.random.choice(np.arange(n), size=n, replace=False)
        return x[shuffle], y[shuffle]

    def _load_array(self, files, x_or_y):
        a = np.load(files[x_or_y], allow_pickle=True)['data_array'].astype(np.float32)
        if x_or_y == "y":
            a = a.T
        a = torch.tensor(a)
        a = a.to(self.device)
        return a

    def _get_current_files(self):
        data = self.data[self.mode]
        filename = data.file_names[self.i]
        return data.files[filename]

    def change_mode(self, mode):
        random.shuffle(self.data["train"].file_names)
        self.mode = mode
        self.i = 0
        self.X_data, self.y_data = self._load_data()

    def _grab_batch(self):
        batch = (self.X_data[:self.batch_size], self.y_data[:self.batch_size].to(self.device))
        self.X_data = self.X_data[self.batch_size:]
        self.y_data = self.y_data[self.batch_size:]
        return batch

    def get_batch(self):
        if len(self.X_data) >= 1:
            return self._grab_batch()
        elif self.i + 1 < self.data[self.mode].n:
            self.i += 1
            self.X_data, self.y_data = self._load_data()
            return self._grab_batch()
        else:
            return None, None
