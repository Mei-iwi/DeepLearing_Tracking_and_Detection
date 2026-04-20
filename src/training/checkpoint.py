import torch

'''
     Lưu checkpoit để có thể train tiếp hoặc dùng model tốt nhất
'''
def save_checkpoint(model, optimizer, epoch, best_metric, save_path):
    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)


'''
    Nạp checkpoint vào model và optimizer
    Trả về
        - model
        - optimizer
        - start_epoch
        - best_metric
'''
def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint['epoch'] + 1
    best_metric = checkpoint['best_metric']

    return model, optimizer, start_epoch, best_metric