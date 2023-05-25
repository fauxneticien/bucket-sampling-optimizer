# Try out different samplers for speech data with PyTorch Lightning

Sampling/batching variable-length speech data can be a bit tricky and there are a few libraries dedicated to help out (e.g. [Lhotse](https://github.com/lhotse-speech/lhotse), SpeechBrain's [data-io](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.dataio.dataio.html)).
This repository includes some minimal boilerplate to help test using samplers from some of these libraries to make sure they will play nice with PyTorch Lightning 2.x.

## Bucketing Sampler

Trying out SpeechBrain's `DynamicBatchSampler` and Lhotse's `BucketingSampler` to minimize padding (see diagram below, from https://www.kaggle.com/code/sajjadayobi360/dynamic-padding-sortish-bathes)

<img src="https://s2.uupload.ir/files/79ki_dynamic-pad.png">
