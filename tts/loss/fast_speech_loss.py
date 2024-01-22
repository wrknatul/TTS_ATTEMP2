import torch
import torch.nn as nn

from tts.logger import logger


class FastSpeechLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        Construct loss for FastSpeech2 model
        """
        super().__init__(**kwargs)
        self.mse = nn.MSELoss()

    def __call__(self, mel_pred, mel_target,
                 length_pred, length_target,
                 pitch_pred, pitch_target,
                 energy_pred, energy_target,
                 **kwargs):
        mel_loss = self.mse(mel_pred, mel_target)
        duration_predictor_loss = self.mse(length_pred, torch.log1p(length_target.float()))
        pitch_predictor_loss = self.mse(pitch_pred, torch.log1p(pitch_target.float()))
        energy_predictor_loss = self.mse(energy_pred, torch.log1p(energy_target.float()))
        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss
