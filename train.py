import torch
import lightning.pytorch as pl

from torch.distributed import get_rank
from torch.utils.data import get_worker_info

from boilerplate import LibrisDataModule, DummyModel

# Folder with LibriSpeech/test-clean
LIBRISPEECH_PATH="/john2/scr1/nsan-speech-data/"

class DummyModelForLoggingStuff(DummyModel):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        # See spec of what's returned by collate_fn function in libris.py
        padded_wavs, transcriptions, wav_lengths, utterance_ids, source = batch

        percent_padding = (sum(padded_wavs.size(1) - wav_lengths) / torch.numel(padded_wavs)) * 100
        percent_padding = percent_padding.item()

        # DDP info
        rank = get_rank()
        worker = get_worker_info()
        
        print(f"{rank=},{batch_idx=},{percent_padding=}")

        return None

class VanillaDataModule(LibrisDataModule):

    def __init__(self):
        self.dataset_path = LIBRISPEECH_PATH
        super().__init__(dataset_path=self.dataset_path)

    def train_dataloader(self):

        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=10,
            collate_fn=self._collate_fn,
            shuffle=False,
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
    # use_distributed_sampler=False,
    max_epochs=1,
    # max_steps=1
)

trainer.fit(
    DummyModelForLoggingStuff(),
    datamodule=VanillaDataModule()
)

