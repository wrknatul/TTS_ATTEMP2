import numpy as np
import random
import os
import pyworld as pw
import scipy.interpolate as interpolate
import time
import torch
import torchaudio
from tqdm import tqdm


from tts.utils import ROOT_PATH
from tts.logger import logger
from tts.utils.text import text_to_sequence


class LJSpeechDataset(object):
    def __init__(self, text_cleaners=("english_cleaners",), limit=None):
        self._data_dir = ROOT_PATH / "data"
        self._text_cleaners = text_cleaners
        self.index = self._load_index()
        if limit is not None:
            random.seed(42)
            random.shuffle(self.index)
            self.index = self.index[:limit]

    def _load_index(self):
        text = self.process_text(self._data_dir / "train.txt")
        pitch_dir, energy_dir = self._save_pitch_and_energy(len(text))
        start = time.perf_counter()
        index = []
        for i in tqdm(range(len(text)), desc="Loading index"):
            mel_gt_name = os.path.join(self._data_dir / "mels", "ljspeech-mel-%05d.npy" % (i + 1))
            mel_gt_target = np.load(mel_gt_name)
            duration = np.load(os.path.join(self._data_dir / "alignments", str(i) + ".npy"))
            character = text[i][0:len(text[i]) - 1]
            character = np.array(text_to_sequence(character, self._text_cleaners))
            pitch_gt_path = os.path.join(pitch_dir, "ljspeech-pitch-%05d.npy" % (i + 1))
            pitch_gt_target = np.load(pitch_gt_path).astype(np.float32)
            energy_gt_path = os.path.join(energy_dir, "ljspeech-energy-%05d.npy" % (i + 1))
            energy_gt_target = np.load(energy_gt_path)

            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            mel_gt_target = torch.from_numpy(mel_gt_target)
            pitch_gt_target = torch.from_numpy(pitch_gt_target)
            energy_gt_target = torch.from_numpy(energy_gt_target)

            index.append(
                {
                    "text": character,
                    "duration_target": duration,
                    "mel_target": mel_gt_target,
                    "pitch_target": pitch_gt_target,
                    "energy_target": energy_gt_target
                }
            )

        logger.info(f"Cost {time.perf_counter() - start:.2f}s to load all data.")

        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        return self.index[item]

    def _save_pitch_and_energy(self, length: int):
        pitch_dir = self._data_dir / "pitch"
        energy_dir = self._data_dir / "energy"
        if pitch_dir.exists() and energy_dir.exists() and len(os.listdir(str(pitch_dir))) >= length:
            return pitch_dir, energy_dir
        pitch_dir.mkdir(exist_ok=True, parents=True)
        energy_dir.mkdir(exist_ok=True, parents=True)

        wav_path = self._data_dir / "LJSpeech-1.1" / "wavs"
        mel_path = self._data_dir / "mels"
        min_pitch, max_pitch = 1e9, -1e9
        min_energy, max_energy = 1e9, -1e9
        for idx, wav_p in enumerate(tqdm(sorted(wav_path.iterdir()), desc="Saving pitch and energy")):
            mel = np.load(os.path.join(mel_path, "ljspeech-mel-%05d.npy" % (idx + 1)))

            # energy
            energy = np.linalg.norm(mel, axis=-1)
            np.save(os.path.join(energy_dir, "ljspeech-energy-%05d.npy" % (idx + 1)), energy)

            min_energy = min(min_energy, energy.min())
            max_energy = max(max_energy, energy.max())

            # pitch
            wave, sr = torchaudio.load(wav_p)
            wave = wave.to(torch.float64).numpy().sum(axis=0)

            frame_period = (wave.shape[0] / sr * 1000) / mel.shape[0]
            _f0, t = pw.dio(wave, sr, frame_period=frame_period)
            f0 = pw.stonemask(wave, _f0, t, sr)[:mel.shape[0]]

            idx_nonzero = np.nonzero(f0)
            x = np.arange(f0.shape[0])[idx_nonzero]
            values = (f0[idx_nonzero][0], f0[idx_nonzero][-1])
            f = interpolate.interp1d(x, f0[idx_nonzero], bounds_error=False, fill_value=values)
            new_f0 = f(np.arange(f0.shape[0]))

            np.save(os.path.join(pitch_dir, "ljspeech-pitch-%05d.npy" % (idx + 1)), new_f0)

            min_pitch = min(min_pitch, new_f0.min())
            max_pitch = max(max_pitch, new_f0.max())

        logger.info(f"Pitch:\n\tMin: {min_pitch:.4f}\n\tMax: {max_pitch:.4f}")
        logger.info(f"Energy:\n\tMin: {min_energy:.4f}\n\tMax: {max_energy:.4f}")

        return pitch_dir, energy_dir

    @staticmethod
    def process_text(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            txt = []
            for line in f.readlines():
                txt.append(line)
            return txt
