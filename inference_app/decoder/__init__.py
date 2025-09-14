"""
Beam Search Decoding Package for CTC Models

This package provides beam search decoding functionality for CTC-based models,
particularly designed for handwriting recognition tasks. It includes support
for KenLM language models to improve decoding accuracy.

Features:
- KenLM language model integration (ARPA and binary formats)
- Weighted beam search decoding with temperature scaling
- CTC decoder setup with configurable parameters
- Fallback to greedy decoding on errors
- Support for batch processing

Modules:
    setup: Functions for setting up beam search decoders with KenLM models
    wbs: Weighted beam search decoding functions

Dependencies:
    - pyctcdecode: For CTC beam search decoding
    - torch: For tensor operations
    - numpy: For numerical computations
    - KenLM: Language modeling (installed via pyctcdecode)

Example usage:
    from beam_search import setup_decoder, wbs_decode
    
    # Setup decoder with language model
    vocab = ['<BLANK>'] + list("abcdefghijklmnopqrstuvwxyz ")
    decoder = setup_decoder(vocab, "path/to/language_model.arpa")
    
    # Decode predictions
    predictions = wbs_decode(
        log_probs_btV=model_output,
        decoder=decoder,
        temperature=1.1
    )
"""

# Import main functions
from .setup import setup_decoder
from .wbs import wbs_decode

# Define what gets imported with "from beam_search import *"
__all__ = [
    "setup_decoder",
    "wbs_decode"
]


# Default vocabulary for handwriting recognition
DEFAULT_VOCAB = [
    '<BLANK>'
] + list("!#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")

# Default decoder parameters
DEFAULT_DECODER_PARAMS = {
    'alpha': 0.35,           # Language model weight
    'beta': 0.6,             # Word insertion bonus
    'unk_score_offset': -10.0,  # Unknown word penalty
    'temperature': 1.1,      # Temperature scaling for softmax
    'blank_is_first': True   # Whether blank token is at index 0
}

def quick_setup(model_path, vocab=None, **kwargs):
    """
    Quick setup function for common use cases.
    
    Args:
        model_path (str): Path to KenLM language model
        vocab (list, optional): Vocabulary list. Defaults to DEFAULT_VOCAB
        **kwargs: Additional parameters for decoder setup
    
    Returns:
        decoder: Configured CTC decoder
    """
    if vocab is None:
        vocab = DEFAULT_VOCAB
    
    return setup_decoder(vocab, model_path, **kwargs)

# Add quick_setup to exports
__all__.append("quick_setup")
__all__.extend(["DEFAULT_VOCAB", "DEFAULT_DECODER_PARAMS"])
