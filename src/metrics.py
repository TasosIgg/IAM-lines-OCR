import editdistance

def word_edit_distance(pred_words, gt_words):
    """Calculate edit distance between word sequences."""
    m, n = len(pred_words), len(gt_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == gt_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def calculate_accuracy(predictions, ground_truths):
    """Calculate character and word accuracy."""
    preds = [str(p) for p in predictions]
    gts = [str(g) for g in ground_truths]
    
    if not preds or not gts or len(preds) != len(gts):
        return 0.0, 0.0
    
    # Character accuracy
    total_char_errors = sum(editdistance.eval(p, g) for p, g in zip(preds, gts))
    total_gt_chars = sum(len(g) for g in gts)
    
    if total_gt_chars == 0:
        char_accuracy = 1.0 if total_char_errors == 0 else 0.0
    else:
        cer = total_char_errors / total_gt_chars
        char_accuracy = max(0.0, 1.0 - cer)
    
    # Word accuracy 
    total_word_errors = 0
    total_gt_words = 0
    for pred, gt in zip(preds, gts):
        pred_words = pred.split()
        gt_words = gt.split()
        total_word_errors += word_edit_distance(pred_words, gt_words)
        total_gt_words += len(gt_words)
    
    if total_gt_words == 0:
        word_accuracy = 1.0 if total_word_errors == 0 else 0.0
    else:
        wer = total_word_errors / total_gt_words
        word_accuracy = max(0.0, 1.0 - wer)
    
    return char_accuracy, word_accuracy

def calculate_edit_distance(predictions, ground_truths):
    """Calculate normalized edit distance."""
    total_distance = 0
    total_length = 0
    
    for pred, gt in zip(predictions, ground_truths):
        distance = editdistance.eval(pred, gt)
        total_distance += distance
        total_length += max(len(pred), len(gt), 1)
    
    return total_distance / max(total_length, 1)
