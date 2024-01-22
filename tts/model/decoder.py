import torch.nn as nn

from tts.utils.constants import PAD
from .fft import FFTBlock
from .utils import get_attn_key_pad_mask, get_non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 encoder_dim,
                 max_seq_len,
                 dropout,
                 head,
                 conv1d_filter_size,
                 n_layers,
                 **fft_params):
        super().__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD,
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

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, pad=PAD)
        non_pad_mask = get_non_pad_mask(enc_pos, pad=PAD)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
