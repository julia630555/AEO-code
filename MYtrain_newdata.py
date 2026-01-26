import os
import json
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ===============================
# 1. 配置与种子
# ===============================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# ===============================
# 2. 数据读取工具 (支持多路径)
# ===============================
def read_hd_data(paths):
    """
    支持传入一个路径列表。如果路径是目录，则读取目录下所有json；如果是文件，则直接读取。
    拼接 hd 和 hd3 特征。
    """
    items = []
    files = []
    
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist. Skipping.")
            continue
        if os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                if fn.endswith(".json"):
                    files.append(os.path.join(path, fn))
        else:
            files.append(path)

    print(f"Loading data from {len(files)} files...")
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
            
        if isinstance(data, dict): data = [data]
        
        for it in data:
            hd = it.get("hd")
            hd3 = it.get("hd3")
            
            # 基础过滤：必须有 hd 和 hd3 特征
            if hd is None or hd3 is None:
                continue
            
            # 解析标签
            lab = None
            if "response_label" in it:
                rl = it["response_label"].lower()
                if "violate" in rl:
                    lab = 1
                elif "follow" in rl:
                    lab = 0
            
            if lab is None and "label" in it:
                ll = str(it["label"]).lower()
                if ll in ["conflict", "violate", "1"]:
                    lab = 1
                elif ll in ["normal", "ok", "0"]:
                    lab = 0
            
            if lab is not None:
                # 拼接 hd 和 hd3 (4096 + 4096 = 8192)
                combined_vec = np.array(hd + hd3, dtype=np.float32)
                items.append({
                    "input": combined_vec,
                    "label": lab
                })
                
    return items

# ===============================
# 3. 数据集与其模型定义
# ===============================
class MindDataset(Dataset):
    def __init__(self, data_list):
        # 优化：先转换为单个 numpy 数组，再转为 Tensor，避免 UserWarning 并提升速度
        inputs_np = np.stack([d["input"] for d in data_list])
        self.inputs = torch.tensor(inputs_np, dtype=torch.float32)
        self.labels = torch.tensor([d["label"] for d in data_list], dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class EntropyClassifier(nn.Module):
    def __init__(self, input_size=8192, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ===============================
# 4. 主训练逻辑
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Train MIND Classifier with combined data sources.")
    parser.add_argument("--data_paths", type=str, nargs="+", required=True, help="List of data directories or json files.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the best model.")
    parser.add_argument("--model_name", type=str, default="best_mind_entropy.pt", help="Name of the saved model file.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--training_split", type=float, default=None, help="Fraction of the total data to use for training/validation (default: all).")
    parser.add_argument("--entropy_weight", type=float, default=1.0, help="Weight for entropy maximization on Violate class.")

    args = parser.parse_args()
    
    # 自动识别模型标识逻辑
    model_id = ""
    # 增加 Phi3 到关键字列表
    keywords = ["Mistral", "Llama", "Qwen", "Gemma", "Baichuan", "InternLM", "ChatGLM", "Yi", "Phi3"]
    for dp in args.data_paths:
        # 排除通用的数据目录名，避免如 "Yi" 误匹配 "MyIHEvaldata"
        parts = dp.rstrip(os.sep).split(os.sep)
        for p in reversed(parts):
            if p.lower() == "myihevaldata":
                continue
            if any(kw.lower() in p.lower() for kw in keywords):
                model_id = p
                break
        if model_id:
            break
            
    if model_id and args.model_name == "best_mind_entropy.pt":
        name_base, ext = os.path.splitext(args.model_name)
        args.model_name = f"{name_base}_{model_id}{ext}"
        print(f"Auto-detected model identity: {model_id}")
        print(f"Updated model save name to: {args.model_name}")

    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载并合并所有数据
    print("-" * 30)
    all_items = read_hd_data(args.data_paths)
    if not all_items:
        print("No valid data loaded. Exit.")
        return
    print(f"Total samples collected: {len(all_items)}")

    # 2. 划分训练/验证集
    random.shuffle(all_items)
    
    # 如果指定了 training_split，则只使用部分数据
    if args.training_split is not None:
        num_use = int(len(all_items) * args.training_split)
        all_items = all_items[:num_use]
        print(f"Reduced data to {len(all_items)} samples (split ratio: {args.training_split})")

    # 验证集数量为训练集的 0.1 (即总数的 1/11)
    val_size = int(len(all_items) / 11)
    train_data = all_items[val_size:]
    val_data = all_items[:val_size]
    
    print(f"Dataset sizes -> Train: {len(train_data)}, Valid: {len(val_data)}")

    train_loader = DataLoader(MindDataset(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MindDataset(val_data), batch_size=args.batch_size, shuffle=False)

    # 3. 初始化模型与优化器
    # 自动获取输入维度 (例如 Llama 为 8192, Phi3 为 6144)
    sample_input = train_data[0]["input"]
    actual_input_size = int(sample_input.shape[0])
    print(f"Detected input dimension: {actual_input_size}")
    
    model = EntropyClassifier(input_size=actual_input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 类别权重平衡 (仅在计算 CrossEntropy 时作为辅助)
    labels = [d["label"] for d in train_data]
    count_0 = labels.count(0)
    count_1 = labels.count(1)
    print(f"Train set: Follow={count_0}, Violate={count_1}")

    # 4. 训练循环
    best_acc = 0.0
    best_model_path = os.path.join(args.output_path, args.model_name)

    print("-" * 30)
    print("Training Mode: Entropy Maximization for Violate class (Label 1)")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            
            # 分离两类数据
            mask_0 = (y == 0)
            mask_1 = (y == 1)
            
            loss = 0.0
            
            # 对 Label 0 (Follow): 最小化交叉熵 -> 推向确信 (Low Entropy)
            if mask_0.any():
                logits_0 = logits[mask_0]
                target_0 = y[mask_0]
                loss += F.cross_entropy(logits_0, target_0)
                
            # 对 Label 1 (Violate): 极大化熵 -> 推向迷茫 (High Entropy)
            if mask_1.any():
                logits_1 = logits[mask_1]
                probs_1 = F.softmax(logits_1, dim=1)
                entropy_1 = -torch.sum(probs_1 * torch.log(probs_1 + 1e-10), dim=1).mean()
                # 目标熵为 ln(2) ≈ 0.693
                # 我们希望 entropy_1 靠近 0.693，最小化这个差距
                loss += args.entropy_weight * (0.693 - entropy_1)
            
            if loss == 0.0: continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New Best Model Saved with Acc: {val_acc:.4f} ***")

    print(f"\nTraining Complete. Best Val Acc: {best_acc:.4f}")
    print(f"Model saved to: {best_model_path}")

if __name__ == "__main__":
    main()
