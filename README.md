# OCR Training Results - IAM Handwritten Text Recognition

## Project Overview
CRNN-based handwritten text recognition system trained on the IAM Lines dataset, implementing:
- Convolutional Neural Network with residual blocks and spatial attention
- Bidirectional LSTM layers for sequence modeling
- CTC loss for variable-length sequence alignment
- Beam search decoding with KenLM language model integration

## Final Performance Metrics
- **Character Error Rate (CER)**: 4.6% 
- **Word Error Rate (WER)**: 14.1%
- **Normalized Edit Distance**: 0.046
- **Average Edit Distance per Sample**: 1.93 characters

## Training Configuration
- **Total Epochs**: 20 (early stopping triggered)
- **Best Model**: Epoch 14
- **Batch Size**: 8
- **Learning Rate**: 3e-4 (with CosineAnnealingWarmRestarts)
- **Architecture**: CRNN with spatial attention, 512 hidden units, 2 LSTM layers
- **Preprocessing**: Adaptive Gaussian thresholding, contrast enhancement, augmentation

## Training Progress

| Epoch | Train Loss | Val Loss | Train Char Acc | Val Char Acc | Train Word Acc | Val Word Acc |
|-------|------------|----------|----------------|--------------|----------------|--------------|
| 1     | 1.7005     | 0.6599   | 0.533          | 0.879        | 0.397          | 0.723        |
| 2     | 0.6492     | 0.4190   | 0.900          | 0.933        | 0.790          | 0.815        |
| 3     | 0.4462     | 0.3291   | 0.944          | 0.947        | 0.862          | 0.840        |
| 4     | 0.3323     | 0.2778   | 0.958          | 0.957        | 0.893          | 0.860        |
| 5     | 0.5511     | 0.3866   | 0.925          | 0.947        | 0.827          | 0.843        |
| 6     | 0.4857     | 0.3796   | 0.936          | 0.939        | 0.850          | 0.832        |
| 7     | 0.4253     | 0.3575   | 0.948          | 0.944        | 0.876          | 0.836        |
| 8     | 0.3675     | 0.3055   | 0.958          | 0.954        | 0.895          | 0.866        |
| 9     | 0.3191     | 0.2914   | 0.961          | 0.956        | 0.898          | 0.858        |
| 10    | 0.2704     | 0.2536   | 0.966          | 0.964        | 0.915          | 0.878        |
| 11    | 0.2194     | 0.2415   | 0.974          | 0.963        | 0.933          | 0.876        |
| 12    | 0.1840     | 0.2161   | 0.979          | 0.968        | 0.943          | 0.887        |
| 13    | 0.1506     | 0.1997   | 0.984          | 0.972        | 0.956          | 0.899        |
| **14**| **0.1354** | **0.1913** | **0.986**    | **0.972**    | **0.955**      | **0.898**    |
| 15    | 0.3416     | 0.2940   | 0.962          | 0.958        | 0.905          | 0.860        |
| 16    | 0.3374     | 0.3139   | 0.963          | 0.952        | 0.906          | 0.853        |
| 17    | 0.3093     | 0.3001   | 0.962          | 0.951        | 0.905          | 0.845        |
| 18    | 0.2927     | 0.2837   | 0.965          | 0.959        | 0.912          | 0.863        |
| 19    | 0.2698     | 0.2880   | 0.970          | 0.961        | 0.924          | 0.868        |
| 20    | 0.2569     | 0.2697   | 0.971          | 0.960        | 0.925          | 0.863        |

## Key Training Observations

### Learning Dynamics
- **Rapid initial improvement**: CER dropped from 46.7% to 12.1% in first epoch
- **Consistent progress**: Steady improvement through epoch 14
- **Early stopping**: Triggered after 6 epochs without validation improvement
- **Learning rate scheduling**: CosineAnnealingWarmRestarts with T_0=5, T_mult=2

### Model Convergence
- **Best validation loss**: 0.1913 at epoch 14 (ctc loss)
- **Training time per epoch**: ~6-10 minutes (decreasing over time)
- **Overfitting signs**: Small divergence between train/val metrics in later epochs
- **Generalization**: Strong test performance (95.4% char accuracy) indicates good generalization

## Architecture Details
- **Input**: 128×1028 grayscale images
- **CNN Backbone**: Residual blocks with GELU activation
- **Attention**: Multi-head spatial attention (4 heads)
- **RNN**: 2-layer bidirectional LSTM (512 hidden units each)
- **Output**: CTC-compatible log probabilities over 80-character vocabulary
- **Decoding**: Beam search with 3-gram KenLM language model

## Dataset Statistics
- **Training samples**: 6,161 (after filtering)
- **Validation samples**: 966 (after filtering)  
- **Test samples**: 2,915 (after filtering)
- **Vocabulary size**: 79 characters + blank token
- 
## Technical Implementation Highlights
- **Custom preprocessing pipeline** with multiple binarization options
- **Data augmentation** using Albumentations for training robustness
- **Spatial attention mechanism** to focus on relevant image regions
- **KenLM integration** for linguistically-informed beam search decoding
- **Comprehensive evaluation** with multiple metrics (CER, WER, edit distance)


## How to Run
1. Install dependencies: `pip install datasets editdistance albumentations pyctcdecode kenlm`
2. Open the notebook in Kaggle/Colab or run locally
3. The notebook handles dataset loading, preprocessing, training, and evaluation automatically
