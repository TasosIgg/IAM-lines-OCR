import random
import numpy as np
import torch

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Training hyperparameters
LEARNING_RATE = 3e-4
EPOCHS = 80
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
DROPOUT = 0.2

# Image preprocessing
TARGET_HEIGHT = 128
TARGET_WIDTH = 1028

# Vocabulary
VOCAB = list("!#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
VOCAB = ['<BLANK>'] + VOCAB
VOCAB_SIZE = len(VOCAB)

# Create mappings
CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCAB)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
CHARS = VOCAB[1:]  # Remove <BLANK> for pyctcdecode
