import torch
import numpy as np


def wbs_decode(log_probs_btV, decoder, blank_is_first=True, temperature=1.1, chars=None):
    """Decode batch using beam search with language model"""
    # Apply temperature scaling
    if temperature != 1.0:
        log_probs_btV = log_probs_btV / temperature
        log_probs_btV = torch.log_softmax(log_probs_btV, dim=-1)
    
    # Convert to numpy
    logits = log_probs_btV.detach().cpu().numpy()
    
    # Move blank to last position if needed (pyctcdecode expects blank last)
    if blank_is_first:
        logits = np.concatenate([logits[:, :, 1:], logits[:, :, :1]], axis=2)
    
    texts = []
    for i in range(logits.shape[0]):
        try:
            text = decoder.decode(logits[i])
            texts.append(text)
        except Exception as e:
            print(f"Decode error for batch {i}: {e}")
            # Fallback to greedy decode
            greedy_pred = np.argmax(logits[i], axis=1)
            vocab = ['<BLANK>'] + list("!#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
            # Reconstruct text from predictions (accounting for blank repositioning)
            if blank_is_first:
                # Map back to original vocab indices
                text_chars = []
                for idx in greedy_pred:
                    if idx < len(vocab) - 1:  # Not blank
                        text_chars.append(vocab[idx + 1])  # +1 because we removed blank
                text = ''.join(text_chars)
            else:
                text = ''.join([vocab[idx] if idx < len(vocab) else '' for idx in greedy_pred])
            texts.append(text)
    
    return texts
