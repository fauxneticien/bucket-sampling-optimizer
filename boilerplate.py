# Based on discussion at https://github.com/Lightning-AI/lightning/issues/15734
# And notebook provided by Lightning engineer Adrian Wälchli (https://github.com/awaelchli)
# https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-

import torch
import torchaudio
import os
import tqdm

import torch.nn as nn
import lightning.pytorch as pl

from torch.distributed import get_rank
from torch.utils.data import get_worker_info
from torch.optim import Adam
from typing import List

# From https://github.com/pytorch/audio/blob/c6624fa6aa0154f5968b4bb40bd57c56349c41d7/examples/hubert/dataset/hubert_dataset.py#L446
def _get_lengths_librispeech(files: List[str], path: str, ext_audio: str) -> List[int]:
    lengths = []
    print("Getting LibriSpeech durations ...")
    for file_path in tqdm.tqdm(files):
        speaker_id, chapter_id, utterance_id = file_path.split("-")
        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + ext_audio
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
        length = torchaudio.info(file_audio).num_frames
        lengths.append(length)
    return lengths

class LibrisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str
    ):
        super().__init__()

        self.dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_path, "test-clean")
        self.dataset_lengths = _get_lengths_librispeech(self.dataset._walker, self.dataset._path, self.dataset._ext_audio)

    def _collate_fn(self, data):
        waveforms, transcriptions, wav_lengths, utterance_ids = [], [], [], []

        for (waveform, _, transcription, speaker_id, chapter_id, utterance_id) in data:

            waveforms.append(waveform)
            transcriptions.append(transcription)

            wav_lengths.append(waveform.size(1))

            full_id = f"{speaker_id}-{chapter_id}-{utterance_id}"
            utterance_ids.append(full_id)

        padded_wavs = torch.zeros(len(data), max(wav_lengths))
        for i in range(len(waveforms)):
            padded_wavs[i][0 : waveforms[i].shape[1]] = waveforms[i]

        wav_lengths = torch.tensor(wav_lengths)

        # Future-proof collator return specification
        # for when we want to try mixing multiple datasets
        source = "librispeech"

        return padded_wavs, transcriptions, wav_lengths, utterance_ids, source
    
class DummyModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def configure_optimizers(self):
        # required by Trainer, but not relevant for this test
        return Adam(self.parameters())
