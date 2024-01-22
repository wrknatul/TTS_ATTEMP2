import numpy as np
import torch
import torch.nn.functional as F
from typing import List


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len - x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def process_lengths(arr):
    length = np.array([])
    for el in arr:
        length = np.append(length, el.size(0))
    pos = list()
    max_len = int(max(length))
    for length_src_row in length:
        pos.append(np.pad([i + 1 for i in range(int(length_src_row))],
                          (0, max_len - int(length_src_row)), 'constant'))
    return max_len, torch.from_numpy(np.array(pos))


def reprocess_tensor(batch: List[dict], cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration_target"] for ind in cut_list]
    pitches = [batch[ind]["pitch_target"] for ind in cut_list]
    energies = [batch[ind]["energy_target"] for ind in cut_list]

    _, src_pos = process_lengths(texts)
    max_mel_len, mel_pos = process_lengths(mel_targets)

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    pitch_targets = pad_1D_tensor(pitches)
    energy_targets = pad_1D_tensor(energies)

    return {
        "src_seq": texts,
        "length_target": durations,
        "mel_pos": mel_pos,
        "mel_target": mel_targets,
        "src_pos": src_pos,
        "mel_max_len": max_mel_len,
        "pitch_target": pitch_targets,
        "energy_target": energy_targets
    }
