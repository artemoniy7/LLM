# LLM

## Checkpoints

Training now saves model weights to `model.ckpt`. If the file already exists, `train.cpp` loads it and continues training instead of starting from random weights, so progress is not lost between runs. The chat executable uses the same architecture settings and loads `model.ckpt` automatically before generation.
