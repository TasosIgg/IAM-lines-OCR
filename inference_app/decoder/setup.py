def setup_decoder(vocab, model_path):
    """Setup beam search decoder using pre-trained KenLM model"""
    import os
    from pyctcdecode import build_ctcdecoder
    
    # Verify model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"KenLM model not found at: {model_path}")
    
    # Check file extension to determine model type
    if model_path.endswith('.arpa'):
        print(f"Loading ARPA language model from: {model_path}")
    elif model_path.endswith('.bin'):
        print(f"Loading binary language model from: {model_path}")
    else:
        print(f"Warning: Unknown model format for: {model_path}")
    
    # Build decoder with pre-trained model
    print("Building CTC decoder with pre-trained language model...")
    chars = vocab[1:]  # Remove <BLANK> token for pyctcdecode
    
    decoder = build_ctcdecoder(
        labels=chars,
        kenlm_model_path=model_path,
        alpha=0.35,          # LM weight
        beta=0.6,            # word insertion bonus
        unk_score_offset=-10.0,
    )
    
    print("Beam search decoder ready with pre-trained model!")
    return decoder
