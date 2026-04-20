import torch


'''
    Tính accuracy từ logits và nhãn thật
'''
def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

'''
    Tạo confusion matrix từ nhãn thật và nhãn dự đoán
'''
def compute_confusion_matrix(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().tolist()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().tolist()
    
    num_classes = max(max(y_true), max(y_pred)) + 1
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

'''
    Tín precision, recall, F1_score theo macro average
'''
def compute_precision_recall_f1(y_true, y_pred):
    cm = compute_confusion_matrix(y_true, y_pred)
    num_classes = cm.size(0)

    per_class = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        tp = cm[i, i].item
        fp = cm[:, i].sum().item() - tp #type: ignore
        fn = cm[i, :].sum().item() - tp #type: ignore

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = ((2 * precision * recall) / (precision + recall)  if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        per_class.append({
            "class_id": i,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        return {
            "precision": sum(precisions) / num_classes if num_classes > 0 else 0.0,
            "recall" : sum(recalls) / num_classes if num_classes > 0 else 0.0,
            "f1": sum(f1_scores) / num_classes if num_classes > 0 else 0.0,
            "per_class": per_class
        }