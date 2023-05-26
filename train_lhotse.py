import torch
import lightning.pytorch as pl

from torch.distributed import get_rank
from torch.utils.data import get_worker_info

from boilerplate import LibrisDataModule, DummyModel

# Folder with LibriSpeech/test-clean
LIBRISPEECH_PATH_FOR_LHOTSE="/john2/scr1/nsan-speech-data/LibriSpeech"

class DummyModelForLoggingStuff(DummyModel):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        # For specification of what is returned by K2SpeechRecognitionDataset, see:
        # https://github.com/lhotse-speech/lhotse/blob/a3e78830f248e676e32a51bc0f14160207a61578/lhotse/dataset/speech_recognition.py#L28
        inputs, supervisions = batch['inputs'], batch['supervisions']

        percent_padding = (sum(inputs.size(1) - supervisions['num_samples']) / torch.numel(inputs)) * 100
        percent_padding = percent_padding.item()

        # DDP info
        rank = get_rank()
        worker = get_worker_info()
        
        print(f"{rank=},{batch_idx=},{percent_padding=:.2f}")

        return None

from lhotse.recipes import prepare_librispeech
from lhotse import CutSet
from lhotse.dataset import BucketingSampler, K2SpeechRecognitionDataset
from lhotse.dataset.input_strategies import AudioSamples

libri = prepare_librispeech(corpus_dir=LIBRISPEECH_PATH_FOR_LHOTSE)
cuts = CutSet.from_manifests(**libri['test-clean'])
dataset = K2SpeechRecognitionDataset(cuts, input_strategy=AudioSamples())
# sampler class should be DDP-compatible, according to:
# https://github.com/lhotse-speech/lhotse/blob/a3e78830f248e676e32a51bc0f14160207a61578/lhotse/dataset/sampling/base.py#L24
sampler = BucketingSampler(cuts, max_duration=300, shuffle=True)

dataloader = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=None,
    num_workers=8
)

trainer = pl.Trainer(
    accelerator="cpu", 
    devices=1,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
    logger=False,
    strategy="ddp",
    # Prevent Lightning from replacing Lhotse's DDP-compatible sampler
    use_distributed_sampler=False,
    # max_epochs=1,
    max_steps=1
)

trainer.fit(
    DummyModelForLoggingStuff(),
    dataloader
)
