import os
import tempfile
import re
import numpy as np
import torch
from pyctcdecode import build_ctcdecoder
import kenlm

def setup_kenlm():
    """Setup KenLM for language modeling."""
    print("Installing dependencies...")
    os.system("apt-get update -qq")
    os.system("apt-get install -y build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev")
    
    print("Cloning and building KenLM...")
    if not os.path.exists("/kaggle/working/kenlm"):
        os.system("git clone https://github.com/kpu/kenlm.git")
    os.system("cd kenlm && mkdir -p build && cd build && cmake .. && make -j$(nproc)")
    
    kenlm_path = "/kaggle/working/kenlm/build/bin"
    lmplz_path = os.path.join(kenlm_path, "lmplz")
    
    if os.path.exists(lmplz_path):
        print(f"lmplz found at: {lmplz_path}")
        return lmplz_path
    else:
        print("lmplz not found.")
        return None

def clean_text_for_kenlm(text):
    """Clean text for KenLM compatibility."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove problematic tokens
    text = text.replace('<unk>', ' ')
    text = text.replace('<UNK>', ' ')
    text = text.replace('<s>', ' ')
    text = text.replace('</s>', ' ')
    text = text.replace('<pad>', ' ')
    text = text.replace('<PAD>', ' ')
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Keep only vocabulary characters
    allowed_chars = set("!#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
    text = ''.join(c for c in text if c in allowed_chars)
    
    return text

def create_kenlm_from_corpus(corpus_text, order=3, lmplz_path=None):
    """Create KenLM model from text corpus."""
    corpus_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    model_file = tempfile.NamedTemporaryFile(mode='w', suffix='.arpa', delete=False)
    
    try:
        # Write corpus
        print("Writing corpus to temporary file...")
        corpus_file.write(corpus_text)
        corpus_file.close()
        
        print(f"Corpus file size: {os.path.getsize(corpus_file.name)} bytes")
        
        # Find lmplz executable
        if lmplz_path and os.path.exists(lmplz_path):
            lmplz_cmd = lmplz_path
        else:
            possible_paths = [
                "/kaggle/working/kenlm/build/bin/lmplz",
                "/kaggle/working/kenlm/bin/lmplz",
                "lmplz"
            ]
            
            lmplz_cmd = None
            for path in possible_paths:
                if os.path.exists(path) or path == "lmplz":
                    lmplz_cmd = path
                    break
            
            if lmplz_cmd is None:
                raise FileNotFoundError("lmplz executable not found!")
        
        # Build language model
        cmd = f"{lmplz_cmd} -o {order} --discount_fallback --skip_symbols < {corpus_file.name} > {model_file.name}"
        print(f"Running: {cmd}")
        
        result = os.system(cmd)
        
        if result != 0:
            raise RuntimeError(f"lmplz failed with return code {result}")
        
        if not os.path.exists(model_file.name) or os.path.getsize(model_file.name) == 0:
            raise RuntimeError("lmplz produced empty model file")
        
        print(f"Model created successfully: {model_file.name}")
        print(f"Model file size: {os.path.getsize(model_file.name)} bytes")
        
        return model_file.name
        
    except Exception as e:
        if os.path.exists(corpus_file.name):
            os.unlink(corpus_file.name)
        if os.path.exists(model_file.name):
            os.unlink(model_file.name)
        raise e
    finally:
        if os.path.exists(corpus_file.name):
            os.unlink(corpus_file.name)

def setup_beam_search_decoder(vocab, train_data, wiki_data=None, order=3):
    """Setup beam search decoder with language model."""
    lmplz_path = setup_kenlm()
    if lmplz_path is None:
        raise RuntimeError("Failed to setup KenLM")
    
    print("Preparing text corpus...")
    
    # Process IAM texts
    iam_texts = []
    for item in train_data:
        if item.get("text"):
            cleaned = clean_text_for_kenlm(item["text"])
            if cleaned:  
                iam_texts.append(cleaned)
    
    print(f"Cleaned IAM texts: {len(iam_texts)} samples")
    
    # Process WikiText if available
    if wiki_data:
        print("Processing WikiText data...")
        wiki_texts = []
        for _, x in enumerate(wiki_data):
            if x.get('text') and x['text'].strip():
                cleaned = clean_text_for_kenlm(x['text'])
                if cleaned: 
                    wiki_texts.append(cleaned)
        
        print(f"Cleaned WikiText: {len(wiki_texts)} samples")
        combined_texts = iam_texts + wiki_texts[:100000]
    else:
        combined_texts = iam_texts
    
    # Create corpus
    corpus_text = "\n".join(combined_texts)
    print(f"Final corpus: {len(combined_texts)} lines, {len(corpus_text)} characters")
    
    # Create KenLM model
    print(f"Training {order}-gram language model...")
    model_path = create_kenlm_from_corpus(corpus_text, order=order, lmplz_path=lmplz_path)
    
    # Build decoder
    print("Building CTC decoder...")
    chars = vocab[1:]  # Remove <BLANK> token
    decoder = build_ctcdecoder(
        labels=chars,
        kenlm_model_path=model_path,
        alpha=0.35,
        beta=0.6,
        unk_score_offset=-10.0,
    )
    
    print("Beam search decoder ready!")
    return decoder, model_path

def wbs_decode_batch(log_probs_btV, decoder, blank_is_first=True, temperature=1.1, chars=None):
    """Decode batch using beam search with language model."""
    # Apply temperature scaling
    if temperature != 1.0:
        log_probs_btV = log_probs_btV / temperature
        log_probs_btV = torch.log_softmax(log_probs_btV, dim=-1)
    
    # Convert to numpy
    logits = log_probs_btV.detach().cpu().numpy()
    
    # Move blank to last position if needed
    if blank_is_first:
        logits = np.concatenate([logits[:, :, 1:], logits[:, :, :1]], axis=2)
    
    texts = []
    for i in range(logits.shape[0]):
        try:
            text = decoder.decode(logits[i])
            texts.append(text)
        except Exception as e:
            raise RuntimeError(f"Beam search decode failed for batch item {i}: {e}")
    
    return texts
