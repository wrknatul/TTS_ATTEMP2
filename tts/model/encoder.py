import torch.nn as nn

from tts.utils.constants import PAD
from .fft import FFTBlock
from .utils import get_attn_key_pad_mask, get_non_pad_mask


class Encoder(nn.Module):

    def __init__(self,
                 encoder_dim,
                 vocab_size,
                 max_seq_len,
                 dropout,
                 head,
                 conv1d_filter_size,
                 n_layers,
                 **fft_params):
        super().__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            conv1d_filter_size,
            head,
            encoder_dim // head,
            encoder_dim // head,
            dropout=dropout,
            **fft_params
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=PAD)
        non_pad_mask = get_non_pad_mask(src_seq, pad=PAD)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask
