import time
import os
import json
import random
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from metrics import calculate_accuracy, calculate_edit_distance
from beam_search import wbs_decode_batch
import editdistance

def train_epoch(model, train_loader, optimizer, ctc_loss, device, epoch, decoder, chars):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions, all_ground_truths = [], []
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['images'].to(device)
        targets = batch['targets'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        texts = batch['texts']
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  
        outputs_ctc = outputs.permute(1, 0, 2)  
        
        input_lengths = torch.full((images.size(0),), outputs_ctc.size(0), 
                         dtype=torch.long, device=device)
        
        # Calculate CTC loss
        loss = ctc_loss(outputs_ctc, targets, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for metrics (every 10 batches)
        if batch_idx % 10 == 0:
            with torch.no_grad():
                try:
                    preds = wbs_decode_batch(outputs, decoder, blank_is_first=True, chars=chars)
                    all_predictions.extend(preds)
                    all_ground_truths.extend(texts)
                except RuntimeError:
                    pass  # Skip this batch for metrics
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # Calculate training metrics
    avg_loss = total_loss / num_batches
    if all_predictions:
        char_acc, word_acc = calculate_accuracy(all_predictions, all_ground_truths)
        edit_dist = calculate_edit_distance(all_predictions, all_ground_truths)
    else:
        char_acc, word_acc, edit_dist = 0.0, 0.0, 0.0
    
    return avg_loss, char_acc, word_acc, edit_dist

def validate_epoch(model, val_loader, ctc_loss, device, decoder, chars):
    """Validation epoch."""
    model.eval()
    total_loss = 0
    all_predictions, all_ground_truths = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', leave=False):
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            texts = batch['texts']

            # Forward pass
            outputs = model(images)                 
            outputs_ctc = outputs.permute(1, 0, 2) 

            input_lengths = torch.full((images.size(0),), outputs_ctc.size(0), 
                                     dtype=torch.long, device=device)                         

            # Calculate loss
            loss = ctc_loss(outputs_ctc, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # Decode predictions
            try:
                preds = wbs_decode_batch(outputs, decoder, blank_is_first=True, chars=chars)
                all_predictions.extend(preds)
                all_ground_truths.extend(texts)
            except RuntimeError:
                all_predictions.extend([""] * len(texts))
                all_ground_truths.extend(texts)

    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    char_acc, word_acc = calculate_accuracy(all_predictions, all_ground_truths)
    edit_dist = calculate_edit_distance(all_predictions, all_ground_truths)

    return avg_loss, char_acc, word_acc, edit_dist, all_predictions[:10], all_ground_truths[:10]

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_dir):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    save_path = Path(save_dir) / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, save_path)
    return save_path

def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0,0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0,0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0,0].set_title('Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Character Accuracy
    axes[0,1].plot(history['train_char_acc'], label='Train Char Acc', color='blue')
    axes[0,1].plot(history['char_acc'], label='Val Char Acc', color='red')
    axes[0,1].set_title('Character Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Word Accuracy
    axes[1,0].plot(history['train_word_acc'], label='Train Word Acc', color='blue')
    axes[1,0].plot(history['word_acc'], label='Val Word Acc', color='red')
    axes[1,0].set_title('Word Accuracy')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Edit Distance
    axes[1,1].plot(history['train_edit_dist'], label='Train Edit Dist', color='blue')
    axes[1,1].plot(history['edit_dist'], label='Val Edit Dist', color='red')
    axes[1,1].set_title('Edit Distance')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Edit Distance')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def train_model(model, train_loader, val_loader, decoder, chars,
                num_epochs=80, learning_rate=3e-4, weight_decay=1e-4, 
                save_dir='checkpoints', patience=7, min_delta=0):
    """Complete training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    Path(save_dir).mkdir(exist_ok=True)
    
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_char_acc': [], 'train_word_acc': [], 'train_edit_dist': [],
        'char_acc': [], 'word_acc': [], 'edit_dist': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = None
    
    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training
        train_loss, train_char_acc, train_word_acc, train_edit_dist = train_epoch(
            model, train_loader, optimizer, ctc_loss, device, epoch, decoder, chars
        )
        
        # Validation
        val_loss, char_acc, word_acc, edit_dist, pred_samples, gt_samples = validate_epoch(
            model, val_loader, ctc_loss, device, decoder, chars
        )
        
        scheduler.step(epoch + 1)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_char_acc'].append(train_char_acc)
        history['train_word_acc'].append(train_word_acc)
        history['train_edit_dist'].append(train_edit_dist)
        history['char_acc'].append(char_acc)
        history['word_acc'].append(word_acc)
        history['edit_dist'].append(edit_dist)
        
        # Print results
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train - Char: {train_char_acc:.3f} | Word: {train_word_acc:.3f} | Edit: {train_edit_dist:.3f}")
        print(f"Val   - Char: {char_acc:.3f} | Word: {word_acc:.3f} | Edit: {edit_dist:.3f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Show sample predictions
        print("\nSample Predictions:")
        for i in range(min(3, len(pred_samples))):
            print(f"GT:   '{gt_samples[i]}'")
            print(f"Pred: '{pred_samples[i]}'")
            print()
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_dir)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_path = checkpoint_path
            print(f" New best model saved (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement ({epochs_without_improvement}/{patience})")
        
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        print("-" * 80)
    
    # Save training history
    history_path = Path(save_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_history(history, save_path=Path(save_dir) / 'training_plot.png')
    
    print(f"\nTraining completed!")
    print(f"Best model: {best_model_path}")
    print(f"Training history saved: {history_path}")
    
    return history, best_model_path

def evaluate_test_samples(model, dataset, device, decoder, chars, num_samples=10):
    """Evaluate model on test samples."""
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    results = []
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            gt = sample['text']
            output = model(image)  
            pred = wbs_decode_batch(output, decoder, blank_is_first=True, chars=chars)[0]
            
            ed = editdistance.eval(pred, gt)
            
            results.append({
                'prediction': pred, 
                'ground_truth': gt, 
                'edit_distance': ed
            })
            
            all_predictions.append(pred)
            all_ground_truths.append(gt)
    
    # Calculate corpus-level metrics
    char_accuracy, word_accuracy = calculate_accuracy(all_predictions, all_ground_truths)
    edit_distance = calculate_edit_distance(all_predictions, all_ground_truths)
    
    print(f"Corpus Char Accuracy: {char_accuracy:.3f}")
    print(f"Corpus Word Accuracy: {word_accuracy:.3f}")
    print(f"Corpus Edit Distance: {edit_distance:.3f}")
    print(f"Avg Edit Distance per sample: {np.mean([r['edit_distance'] for r in results]):.3f}")
    
    # Add corpus metrics to results
    for result in results:
        result['corpus_char_accuracy'] = char_accuracy
        result['corpus_word_accuracy'] = word_accuracy
    
    return results
