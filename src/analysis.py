import os
import cv2
import tempfile
import logging
import re
from typing import List, Tuple
import numpy as np
from PIL import Image
import editdistance
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt

def setup_environment():
    """Set up environment variables and CUDA settings."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def show_random_samples(dataset, n=5):
    """Display random image samples from dataset."""
    samples = dataset.shuffle(seed=42).select(range(n))
    for i, sample in enumerate(samples):
        plt.figure(figsize=(8, 2))
        plt.imshow(sample["image"], cmap="gray")
        plt.title(f"Text: {sample['text']}")
        plt.axis("off")
        plt.show()

def analyze_dataset(dataset):
    """Perform exploratory data analysis on the dataset."""
    # Character frequency analysis
    char_counts = Counter()
    for sample in dataset:  
        char_counts.update(sample["text"])

    chars, freqs = zip(*char_counts.most_common())
    plt.figure(figsize=(20, 6))
    plt.bar(chars, freqs)
    plt.title("Most Common Characters in Dataset")
    plt.xlabel("Character")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Line length distribution
    line_lengths = [len(sample["text"]) for sample in dataset]
    plt.figure(figsize=(10, 5))
    plt.hist(line_lengths, bins=30, color="skyblue", edgecolor="black")
    plt.title("Distribution of Line Text Lengths")
    plt.xlabel("Number of Characters")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    print(f"Average line length: {np.mean(line_lengths):.2f}")

    # Image size analysis (sample)
    widths, heights = [], []
    for sample in dataset.select(range(500)):  
        w, h = sample["image"].size
        widths.append(w)
        heights.append(h)

    plt.figure(figsize=(10, 4))
    plt.hist(widths, bins=30, alpha=0.7, label="Width")
    plt.hist(heights, bins=30, alpha=0.7, label="Height")
    plt.title("Image Width and Height Distribution (First 500 Samples)")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Average Width: {np.mean(widths):.1f}, Average Height: {np.mean(heights):.1f}")

    # Unique characters
    unique_chars = sorted(set(''.join([s["text"] for s in dataset])))  
    print("Unique Characters Found:")
    print("".join(unique_chars))
    print(f"Total Unique Characters: {len(unique_chars)}")

    # Word frequency
    word_counts = Counter(chain.from_iterable(s["text"].split() for s in dataset))  
    print("Top 10 most common words:")
    print(word_counts.most_common(10))
    print(f"Total unique words: {len(word_counts)}")
