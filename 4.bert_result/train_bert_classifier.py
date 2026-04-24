import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from importlib.machinery import SourceFileLoader
# 直接加载 20_news_data 模块
news_data = SourceFileLoader("news_data", "20_news_data.py").load_module()

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 使用修改后的 20_news_data.py 加载数据
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = news_data.load_and_preprocess_data()
    
    # 使用 'bert-base-uncased' tokenizer 和模型
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 数据令牌化
    print("Tokenizing data...")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=64)
    
    # 创建数据集
    train_dataset = NewsDataset(train_encodings, y_train)
    val_dataset = NewsDataset(val_encodings, y_val)
    test_dataset = NewsDataset(test_encodings, y_test)
    
    # 创建数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载模型
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # 优化器
    optim = AdamW(model.parameters(), lr=5e-5)
    
    # 训练循环
    epochs = 2
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optim.step()
            
        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    # 测试模型
    print("Testing model...")
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 保存模型和结果
    output_dir = "bert_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 分类报告
    target_names = ['alt.atheism', 'soc.religion.christian']
    report = classification_report(test_labels, test_preds, target_names=target_names)
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        
    # 混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], target_names)
    plt.yticks([0, 1], target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (BERT)")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()
    
    # 保存指标和训练历史
    metrics = {
        "test_accuracy": test_acc,
        "confusion_matrix": cm.tolist(),
        "history": history
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 绘制训练历史图
    plt.figure(figsize=(12, 4.5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # 仅绘制验证准确率图
    plt.plot(history["val_acc"], label="val", color="darkorange")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    train()
