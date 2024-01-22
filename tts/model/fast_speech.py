import torch
import torch.nn as nn

from tts.base.base_model import BaseModel
from .decoder import Decoder
from .encoder import Encoder
from .adaptor import VarianceAdaptor
from .regulator import LengthRegulator
from .utils import get_mask_from_lengths


class FastSpeech2(BaseModel):
    """ FastSpeech2 """
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_mels,
                 num_bins,
                 dropout,
                 encoder_dim,
                 decoder_dim,
                 encoder_params,
                 decoder_params,
                 fft_params,
                 regulator_params,
                 pitch_params,
                 energy_params):
        super().__init__()

        self.encoder = Encoder(encoder_dim=encoder_dim,
                               vocab_size=vocab_size,
                               max_seq_len=max_seq_len,
                               dropout=dropout,
                               **encoder_params, **fft_params)
        self.length_regulator = LengthRegulator(encoder_dim=encoder_dim, dropout=dropout, **regulator_params)
        self.decoder = Decoder(encoder_dim=encoder_dim,
                               max_seq_len=max_seq_len,
                               dropout=dropout,
                               **decoder_params, **fft_params)

        self.pitch_adaptor = VarianceAdaptor(encoder_dim=encoder_dim, num_bins=num_bins, **pitch_params)
        self.energy_adaptor = VarianceAdaptor(encoder_dim=encoder_dim, num_bins=num_bins, **energy_params)

        self.mel_linear = nn.Linear(decoder_dim, num_mels)

    @staticmethod
    def mask_tensor(mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def reset_alphas(self, alpha_length, alpha_pitch, alpha_energy):
        self.length_regulator.alpha = alpha_length
        self.pitch_adaptor.alpha = alpha_pitch
        self.energy_adaptor.alpha = alpha_energy

    def forward(self,
                src_seq,
                src_pos,
                mel_pos=None,
                mel_max_length=None,
                length_target=None,
                pitch_target=None,
                energy_target=None,
                **kwargs):
        enc_out, _ = self.encoder(src_seq, src_pos)

        out_reg, length_pred = self.length_regulator(enc_out, length_target, mel_max_length)
        pitch_emb, pitch_pred = self.pitch_adaptor(out_reg, pitch_target)
        energy_emb, energy_pred = self.energy_adaptor(out_reg, energy_target)

        s = out_reg + pitch_emb + energy_emb

        if self.training:
            out = self.decoder(s, mel_pos)
            out = self.mask_tensor(out, mel_pos, mel_max_length)
        else:
            out = self.decoder(s, length_pred)

        return {
            "mel_pred": self.mel_linear(out),
            "length_pred": length_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred
        }
