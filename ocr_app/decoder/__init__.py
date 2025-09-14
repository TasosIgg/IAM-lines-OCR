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
