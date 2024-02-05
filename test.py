import argparse
import multiprocessing
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from tts.datasets import LJSpeechDataset
from tts.logger import WanDBWriter
import tts.model as module_model
from tts.utils import ROOT_PATH
from tts.utils.parse_config import ConfigParser
from tts.utils.text import text_to_sequence
from tts.utils.util import load_waveglow
from train import SEED
from waveglow.inference import get_wav
from glow import WaveGlow

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"
torch.manual_seed(SEED)


def main(args, config):
    logger = config.get_logger("test")
    writer: WanDBWriter = WanDBWriter(config, logger) if args.log_wandb else None

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # read text to test
    texts = LJSpeechDataset.process_text(args.input_text)
    src_seq = [text_to_sequence(text[:-1], ["english_cleaners"]) for text in texts]

    waveglow = load_waveglow(args.waveglow_path, device)
    sr = config["preprocessing"]["sr"]
    alphas = [
        (1, 1, 1), (1, 1, 1.2), (1, 1, 0.8),
        (1, 1.2, 1), (1, 0.8, 1), (1.2, 1, 1),
        (0.8, 1, 1), (1.2, 1.2, 1.2), (0.8, 0.8, 0.8)
    ]
    with torch.no_grad():
        for idx, (text, seq) in tqdm(enumerate(zip(texts, src_seq)), total=len(texts), desc="Texts"):
            length = len(seq)
            seq = torch.tensor(seq).to(device).unsqueeze(0)
            pos = torch.arange(1, length + 1).to(device).unsqueeze(0)

            if writer is not None:
                writer.set_step(step=idx)
                writer.add_text(f"test-text", text)
            for a_idx, (alpha_dur, alpha_pitch, alpha_energy) in enumerate(tqdm(alphas, desc="Alphas")):
                model.reset_alphas(alpha_dur, alpha_pitch, alpha_energy)
                pred = model(src_seq=seq, src_pos=pos)
                audio = get_wav(pred["mel_pred"].transpose(-1, -2), waveglow=waveglow, sampling_rate=sr).unsqueeze(0)

                suf = f"{alpha_dur:.1f}-{alpha_pitch:.1f}-{alpha_energy:.1f}"
                torchaudio.save(Path(args.output) / f"{idx + 1}-{suf}.wav", audio, sample_rate=sr)
                if writer is not None:
                    writer.add_audio(f"test-{suf}-audio", audio, sample_rate=sr)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-i",
        "--input-text",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH.parent / "text.txt"),
        help="Path to file with texts to test"
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-w",
        "--waveglow-path",
        default="waveglow/pretrained_model/waveglow_256channels.pt",
        type=str,
        help="path to pretrained waveglow model"
    )
    args.add_argument(
        "-o",
        "--output",
        default="output",
        type=str,
        help="Dir to write result audio",
    )
    args.add_argument(
        "-l",
        "--log-wandb",
        default=False,
        type=bool,
        help="Save results in wandb or not (wand params are in config file)"
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # prepare output dir
    Path(args.output).mkdir(exist_ok=True, parents=True)

    main(args, config)
