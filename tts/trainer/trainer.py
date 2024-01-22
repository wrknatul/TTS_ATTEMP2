import logging
import random
from pathlib import Path
from random import shuffle

import PIL.Image
import numpy as np
import torch
from torch.cuda.amp import GradScaler
import torchaudio
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tts.base import BaseTrainer
from tts.logger.utils import plot_spectrogram_to_buf
from tts.utils import inf_loop, MetricTracker
from waveglow.inference import get_wav


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            dataloaders,
            batch_expand_dim,
            waveglow,
            batch_accum=1,
            lr_scheduler=None,
            len_epoch=None,
            log_step=50,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.batch_expand_dim = batch_expand_dim
        self.batch_accum = batch_accum
        self.len_loader = len(self.train_dataloader)
        self.waveglow = waveglow
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step

        self.metrics = ["mel_loss", "dur_loss", "pitch_loss", "energy_loss"]
        self.train_metrics = MetricTracker(
            "loss", "grad norm", *self.metrics, writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["src_seq", "src_pos", "mel_pos", "mel_target",
                               "length_target", "pitch_target", "energy_target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for list_batch_idx, list_batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            for cur_idx, batch in enumerate(list_batch):
                batch_idx = list_batch_idx * self.batch_expand_dim + cur_idx
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        idx=batch_idx,
                        metrics=self.train_metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.train_metrics.update("grad norm", self.get_grad_norm())
                if batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.optimizer.param_groups[0]['lr']
                    )
                    self._log_spectrogram(**batch)
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if batch_idx >= self.len_epoch:
                    return last_train_metrics
        log = last_train_metrics

        return log

    def process_batch(self, batch, is_train: bool, idx: int, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()

        # with torch.autocast(device_type=self.device.type):
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        mel_loss, dur_loss, pitch_loss, energy_loss = self.criterion(**batch)
        batch["loss"] = mel_loss + dur_loss + pitch_loss + energy_loss
        batch.update({"mel_loss": mel_loss, "dur_loss": dur_loss, "pitch_loss": pitch_loss, "energy_loss": energy_loss})

        if is_train:
            batch["loss"].backward()
            if (idx - 1) % self.batch_accum == 0:
                self._clip_grad_norm()
                self.optimizer.step()
            if self.lr_scheduler is not None and \
                    not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            try:
                metrics.update(met, batch[met])
            except Exception as err:
                self.logger.warning(f'Caught {err}')
                metrics.update(met, np.nan)
        return batch

    def _log_spectrogram(self, mel_target, mel_pred, **batch):
        idx = np.random.choice(np.arange(len(mel_target)))
        for mels, name in zip([mel_target, mel_pred], ["target_spec", "pred_spec"]):
            img = PIL.Image.open(plot_spectrogram_to_buf(mels[idx].detach().cpu().numpy().T))
            self.writer.add_image(name, ToTensor()(img))
            audio = get_wav(mels[idx].transpose(0, 1).unsqueeze(0),
                            self.waveglow, sampling_rate=self.config["preprocessing"]["sr"])
            self._log_audio(audio, name.replace("spec", "wav"))

    def _log_audio(self, audio, name):
        self.writer.add_audio(name, audio, sample_rate=self.config["preprocessing"]["sr"])

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
