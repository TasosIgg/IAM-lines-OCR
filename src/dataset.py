import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
from preprocessing import preprocess_image, ALBUMENTATIONS_TRANSFORM

class IAMDataset(Dataset):
    """Dataset wrapper for IAM handwritten text recognition."""
    
    def __init__(self, 
                 hf_dataset,
                 char_to_idx: Dict[str, int],
                 idx_to_char: Dict[int, str],
                 target_height: int = 128,
                 target_width: int = 1028,
                 enhance_contrast: bool = False,
                 normalization_range: str = "0_1",
                 max_text_length: Optional[int] = None,
                 augment: bool = False,
                 log_rescaling: bool = True, 
                 binarization=None, 
                 binary_threshold=127,
                 adaptive_threshold_block_size=11, 
                 adaptive_threshold_c=2):
        
        self.dataset = hf_dataset
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.target_height = target_height
        self.target_width = target_width
        self.enhance_contrast = enhance_contrast
        self.normalization_range = normalization_range
        self.max_text_length = max_text_length
        self.augment = augment
        self.log_rescaling = log_rescaling
        self.binarization = binarization
        self.binary_threshold = binary_threshold
        self.adaptive_threshold_block_size = adaptive_threshold_block_size
        self.adaptive_threshold_c = adaptive_threshold_c
            
        # Filter by max text length
        if max_text_length:
            self.dataset = self.dataset.filter(
                lambda x: len(x["text"]) <= max_text_length
            )
            print(f"Filtered dataset to {len(self.dataset)} samples")
        
        # Filter samples that can't be encoded
        self.valid_indices = []
        for idx in range(len(self.dataset)):
            text = self.dataset[idx]["text"]
            if self._can_encode_text(text):
                self.valid_indices.append(idx)
        
        print(f"Valid samples: {len(self.valid_indices)}/{len(self.dataset)}")
    
    def _can_encode_text(self, text: str) -> bool:
        """Check if text can be encoded with current vocabulary."""
        return all(char in self.char_to_idx for char in text)
    
    def _encode_text(self, text: str) -> List[int]:
        """Encode text to indices."""
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]
    
    def _preprocess_image(self, img):
        """Preprocess image with optional augmentation."""
        if isinstance(img, Image.Image):
            img = np.array(img.convert("L"))
        elif len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if self.augment:
            img = ALBUMENTATIONS_TRANSFORM(image=img)["image"]

        return preprocess_image(
            img,
            target_height=self.target_height,
            target_width=self.target_width,
            enhance_contrast=self.enhance_contrast,
            normalization_range=self.normalization_range,
            log_rescaling=self.log_rescaling,
            binarization=self.binarization,
            binary_threshold=self.binary_threshold,
            adaptive_threshold_block_size=self.adaptive_threshold_block_size,
            adaptive_threshold_c=self.adaptive_threshold_c
        )
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.valid_indices[idx]
        sample = self.dataset[real_idx]
        
        image = self._preprocess_image(sample["image"])
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        text = sample["text"]
        target = self._encode_text(text)
        
        return {
            'image': image,
            'text': text,
            'target': torch.tensor(target, dtype=torch.long),
            'target_length': torch.tensor(len(target), dtype=torch.long)
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    targets = [item['target'] for item in batch]
    target_lengths = torch.stack([item['target_length'] for item in batch])
    targets_concat = torch.cat(targets)
    
    return {
        'images': images,
        'texts': texts,
        'targets': targets_concat,
        'target_lengths': target_lengths
    }

def create_dataloaders(train_dataset, val_dataset, batch_size=8, num_workers=4):
    """Create train and validation dataloaders."""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    return train_loader, val_loader
