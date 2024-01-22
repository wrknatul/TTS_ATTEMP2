import numpy as np
import torch
import torch.nn as nn

from .utils import Transpose
from tts.utils.constants import PAD


class Predictor(nn.Module):
    """ Duration/Pitch/Energy Predictor """
    def __init__(self,
                 encoder_dim,
                 predictor_filter_size,
                 predictor_kernel_size,
                 dropout):
        super().__init__()

        self.input_size = encoder_dim
        self.filter_size = predictor_filter_size
        self.kernel = predictor_kernel_size
        self.conv_output_size = predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class VarianceAdaptor(nn.Module):
    def __init__(self, encoder_dim, num_bins, min_bin, max_bin, alpha, **predictor_params):
        super().__init__()
        self.predictor = Predictor(encoder_dim=encoder_dim, **predictor_params)
        bins = torch.linspace(
            np.log1p(min_bin),
            np.log1p(max_bin + 1),
            num_bins
        )
        self.register_buffer("bins", bins)
        self.emb = nn.Embedding(
            num_bins,
            encoder_dim,
            padding_idx=PAD
        )
        self.alpha = alpha

    def forward(self, x, target):
        pred = self.predictor(x)
        if target is None:
            norm = self.alpha * (torch.exp(pred) - 1)
            max_val = self.emb.num_embeddings
            buckets = torch.bucketize(torch.log1p(norm), self.bins)
            buckets = torch.clip(buckets, min=0, max=max_val - 1)
        else:
            buckets = torch.bucketize(torch.log1p(target), self.bins)
        return self.emb(buckets), pred
