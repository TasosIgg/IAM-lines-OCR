from datasets import load_dataset, concatenate_datasets
from config import *
from analysis import setup_environment, show_random_samples, analyze_dataset
from preprocessing import show_preprocessed_samples
from dataset import IAMDataset, create_dataloaders
from model import CRNN
from beam_search import setup_beam_search_decoder
from training import train_model, evaluate_test_samples
import torch

def main():
    """Main training pipeline."""
    
    # Setup environment
    setup_environment()
    
    # Load dataset
    print("Loading IAM dataset...")
    dataset_all = load_dataset("Teklia/IAM-line")
    
    train_hf = dataset_all["train"]
    val_hf = dataset_all["validation"]
    test_hf = dataset_all["test"]
    full_dataset = concatenate_datasets([train_hf, val_hf, test_hf])
    
    print(f"Train split: {len(train_hf)} samples")
    print(f"Val split:   {len(val_hf)} samples") 
    print(f"Test split:  {len(test_hf)} samples")
    print(f"Total samples: {len(full_dataset)}")
    
    # Show vocabulary info
    print(f"Vocab size (incl. blank): {VOCAB_SIZE}")
    print(f"Characters: {''.join(CHARS)}")
    
    # Exploratory data analysis
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    show_random_samples(full_dataset, n=5)
    analyze_dataset(full_dataset)
    
    # Setup beam search decoder
    print("\n=== SETTING UP BEAM SEARCH DECODER ===")
    try:
        print("Loading WikiText...")
        wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train")
        print(f"WikiText loaded: {len(wikitext)} samples")
    except Exception as e:
        wikitext = None
        print(f"WikiText not available: {e}")
    
    try:
        print("Setting up beam search decoder...")
        decoder, model_path = setup_beam_search_decoder(
            vocab=VOCAB, train_data=train_hf, wiki_data=wikitext, order=3
        )
        print("Beam search decoder setup complete!")
    except Exception as e:
        print(f"Setup failed: {e}")
        print("Using basic CTC decoder...")
        from pyctcdecode import build_ctcdecoder
        decoder = build_ctcdecoder(labels=CHARS)
        model_path = None
    
    # Create datasets
    print("\n=== CREATING DATASETS ===")
    train_dataset = IAMDataset(
        train_hf, CHAR_TO_IDX, IDX_TO_CHAR,
        augment=True, enhance_contrast=True, binarization="adaptive_gaussian"
    )
    
    val_dataset = IAMDataset(
        val_hf, CHAR_TO_IDX, IDX_TO_CHAR,
        augment=False, enhance_contrast=True, binarization="adaptive_gaussian"
    )
    
    test_dataset = IAMDataset(
        test_hf, CHAR_TO_IDX, IDX_TO_CHAR,
        augment=False, enhance_contrast=True, binarization="adaptive_gaussian"
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=2
    )
    
    # Show preprocessing examples
    show_preprocessed_samples(full_dataset, n=3)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Test data loading
    print("\nTesting data loading...")
    batch = next(iter(train_loader))
    print(f"Batch images shape: {batch['images'].shape}")
    print(f"Sample text: '{batch['texts'][0]}'")
    
    # Initialize model
    print("\n=== INITIALIZING MODEL ===")
    model = CRNN(
        vocab_size=VOCAB_SIZE, 
        hidden_size=512, 
        num_lstm_layers=2,
        dropout=DROPOUT,
        use_attention=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\n=== STARTING TRAINING ===")
    history, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        decoder=decoder,
        chars=CHARS,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_dir='checkpoints',
        patience=6
    )
    
    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Evaluating on test samples:")
    evaluate_test_samples(model, test_dataset, device, decoder, CHARS, num_samples=100)

if __name__ == "__main__":
    main()
