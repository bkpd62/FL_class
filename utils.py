import os
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_zeroday_accuracy(model, dataloader, device, zeroday_class_index):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            mask = y == zeroday_class_index
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def get_result_filepath(base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"result_{now}.csv")


def save_results_with_metadata(results, args, filepath):
    df = pd.DataFrame(results, columns=["Round", "Global Accuracy", "Zero-day Accuracy", "Z-weight Norm"])

    metadata = {
        "Train Path": args.train_path,
        "Test Path": args.test_path,
        "Num Clients": args.num_clients,
        "Zero-day Client": args.zeroday_client,
        "Exclude": ",".join(args.exclude),
        "LR": args.lr,
        "Epochs": args.epochs,
        "Rounds": args.rounds,
        "Batch Size": args.batch_size,
        "Alpha": args.alpha,
        "Zeroday Round": args.zeroday_round
    }
    meta_df = pd.DataFrame(list(metadata.items()), columns=["Param", "Value"])

    # 저장
    with open(filepath, "w") as f:
        meta_df.to_csv(f, index=False)
        f.write("\n")
        df.to_csv(f, index=False)


def print_confusion_matrix(y_true, y_pred, label_map):
    cm = confusion_matrix(y_true, y_pred)
    labels = [label_map[i] for i in range(len(label_map))]
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))


def expand_model_weights(old_state_dict, client0_state_dict):
    # 기존 모델의 state_dict와 새로운 클래스의 state_dict를 결합하여 확장된 state_dict 생성
    new_state_dict = {}
    for k in old_state_dict:
        if k == 'net.4.weight':
            new_weight = torch.zeros(client0_state_dict[k].shape[0], old_state_dict[k].shape[1])
            new_weight[:old_state_dict[k].shape[0]] = old_state_dict[k]  # 기존 클래스 weight
            new_weight[-1] = client0_state_dict[k][-1]  # 새로운 클래스 weight
            new_state_dict[k] = new_weight
        elif k == 'net.4.bias':
            new_bias = torch.zeros(client0_state_dict[k].shape[0])
            new_bias[:old_state_dict[k].shape[0]] = old_state_dict[k]  # 기존 클래스 bias
            new_bias[-1] = client0_state_dict[k][-1]  # 새로운 클래스 bias
            new_state_dict[k] = new_bias
        else:
            new_state_dict[k] = old_state_dict[k]
    return new_state_dict

def analyze_zeroday_predictions(model, dataloader, device, zeroday_class_index, label_map):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            # zero-day 샘플만 필터링
            mask = (y == zeroday_class_index)
            true_labels.extend(y[mask].cpu().tolist())
            predicted_labels.extend(preds[mask].cpu().tolist())

    # 예측된 라벨의 분포 확인
    from collections import Counter
    pred_counter = Counter(predicted_labels)

    print(f"\n[Analysis] Predictions for zero-day class (Label={zeroday_class_index}, Name={label_map[zeroday_class_index]}):")
    for pred_label, count in pred_counter.items():
        label_name = label_map.get(pred_label, f"Unknown({pred_label})")
        print(f"  → Predicted as {pred_label} ({label_name}): {count} samples")

    return pred_counter


def print_confusion_matrix(model, dataloader, device, label_map):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred)
    labels = [label_map[i] for i in range(len(label_map))]

    print("\n[Confusion Matrix]")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    return cm 
