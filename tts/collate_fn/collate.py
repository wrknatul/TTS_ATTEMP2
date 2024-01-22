import logging
import numpy as np
from typing import List
import torch


from .utils import reprocess_tensor


class Collator(object):
    def __init__(self, batch_expand_dim: int = 32):
        self.batch_expand_dim = batch_expand_dim

    def __call__(self, batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batch_size = len(batch)
        real_batch_size = batch_size // self.batch_expand_dim

        cut_list = []
        for i in range(self.batch_expand_dim):
            cut_list.append(index_arr[i * real_batch_size:(i + 1) * real_batch_size])

        output = []
        for i in range(self.batch_expand_dim):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output
