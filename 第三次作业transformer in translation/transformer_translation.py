import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
from collections import Counter
import re
import os
import os.path as osp

# 简单分词函数
def tokenize_en(text):
    # 简单分词：小写化并按空格分割
    text = text.lower()
    # 移除标点符号
    text = re.sub(r'[\.,!\?"\'\(\)\[\]]', ' ', text)
    # 按空格分割
    tokens = text.split()
    return tokens

def tokenize_fr(text):
    # 简单分词：小写化并按空格分割
    text = text.lower()
    # 移除标点符号
    text = re.sub(r'[\.,!\?"\'\(\)\[\]]', ' ', text)
    # 按空格分割
    tokens = text.split()
    return tokens

# 构建词汇表
class Vocabulary:
    def __init__(self, tokens=None, min_freq=2):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.min_freq = min_freq
        
        if tokens:
            self.build_vocab(tokens)
    
    def build_vocab(self, tokens):
        counter = Counter(tokens)
        idx = 4
        for token, freq in counter.items():
            if freq >= self.min_freq:
                self.itos[idx] = token
                self.stoi[token] = idx
                idx += 1
    
    def __len__(self):
        return len(self.itos)

# 数据集类
class TranslationDataset(Dataset):
    def __init__(self, file_path, src_tokenizer, trg_tokenizer, src_vocab=None, trg_vocab=None, max_len=100, max_samples=None):
        self.src_sentences = []
        self.trg_sentences = []
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_len = max_len
        
        # 读取数据
        # 获取脚本所在目录的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据文件的绝对路径
        abs_file_path = os.path.join(script_dir, file_path)
        print('Reading file:', abs_file_path)
        try:
            with open(abs_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print('Number of lines:', len(lines))
                
                for i, line in enumerate(lines):
                    if max_samples is not None and i >= max_samples:
                        break
                    
                    line = line.strip()
                    if line:
                        try:
                            # 尝试不同的分隔符
                            if '\t' in line:
                                src, trg = line.split('\t', 1)
                            elif '→' in line:
                                src, trg = line.split('→', 1)
                            else:
                                continue
                            self.src_sentences.append(src.strip())
                            self.trg_sentences.append(trg.strip())
                        except Exception as e:
                            if i < 5:  # 只打印前5个错误
                                print('Error processing line', i, ':', e)
            
            print('Number of valid sentences:', len(self.src_sentences))
            
            # 检查是否有有效的句子
            if len(self.src_sentences) == 0:
                raise ValueError('No valid sentences found in the dataset. Please check the file format.')
        except Exception as e:
            print('Error reading file:', e)
            import traceback
            traceback.print_exc()
            # 如果发生异常，并且数据集为空，抛出更详细的错误信息
            if len(self.src_sentences) == 0:
                raise ValueError('Failed to load dataset. Please check the file path and format. Error details: ' + str(e))
        
        # 构建词汇表
        if src_vocab is None:
            all_src_tokens = []
            for sent in self.src_sentences:
                all_src_tokens.extend(self.src_tokenizer(sent))
            self.src_vocab = Vocabulary(all_src_tokens)
        else:
            self.src_vocab = src_vocab
        
        if trg_vocab is None:
            all_trg_tokens = []
            for sent in self.trg_sentences:
                all_trg_tokens.extend(self.trg_tokenizer(sent))
            self.trg_vocab = Vocabulary(all_trg_tokens)
        else:
            self.trg_vocab = trg_vocab
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        trg = self.trg_sentences[idx]
        
        # 分词
        src_tokens = self.src_tokenizer(src)
        trg_tokens = self.trg_tokenizer(trg)
        
        # 添加SOS和EOS标记
        src_tokens = ['<SOS>'] + src_tokens + ['<EOS>']
        trg_tokens = ['<SOS>'] + trg_tokens + ['<EOS>']
        
        # 截断或填充
        if len(src_tokens) > self.max_len:
            src_tokens = src_tokens[:self.max_len]
        else:
            src_tokens += ['<PAD>'] * (self.max_len - len(src_tokens))
        
        if len(trg_tokens) > self.max_len:
            trg_tokens = trg_tokens[:self.max_len]
        else:
            trg_tokens += ['<PAD>'] * (self.max_len - len(trg_tokens))
        
        # 转换为索引
        src_indices = [self.src_vocab.stoi.get(token, self.src_vocab.stoi['<UNK>']) for token in src_tokens]
        trg_indices = [self.trg_vocab.stoi.get(token, self.trg_vocab.stoi['<UNK>']) for token in trg_tokens]
        
        return torch.tensor(src_indices), torch.tensor(trg_indices)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 点积注意力机制
class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        # q: (batch_size, n_heads, seq_len_q, d_k)
        # k: (batch_size, n_heads, seq_len_k, d_k)
        # v: (batch_size, n_heads, seq_len_v, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # 确保mask的形状与scores匹配
            if mask.dim() == 5:
                mask = mask.squeeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        # 确保输出形状正确: (batch_size, n_heads, seq_len_q, d_k)
        return output, attn

# 加性注意力机制
class AdditiveAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(AdditiveAttention, self).__init__()
        self.W_q = nn.Linear(d_k, d_k)
        self.W_k = nn.Linear(d_k, d_k)
        self.v = nn.Linear(d_k, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        # q: (batch_size, n_heads, seq_len_q, d_k)
        # k: (batch_size, n_heads, seq_len_k, d_k)
        # v: (batch_size, n_heads, seq_len_v, d_k)
        
        q = self.W_q(q)
        k = self.W_k(k)
        
        # 计算注意力分数
        # 扩展维度以便广播
        q = q.unsqueeze(3)  # (batch_size, n_heads, seq_len_q, 1, d_k)
        k = k.unsqueeze(2)  # (batch_size, n_heads, 1, seq_len_k, d_k)
        
        # 计算加性注意力分数
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (batch_size, n_heads, seq_len_q, seq_len_k)
        
        if mask is not None:
            # 确保mask的形状与scores匹配
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # (batch_size, n_heads, seq_len_q, d_k)
        return output, attn

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_type='dot', dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(self.n_heads * self.d_k, d_model)
        
        # 注意力机制
        if attention_type == 'dot':
            self.attention = DotProductAttention(dropout)
        elif attention_type == 'additive':
            self.attention = AdditiveAttention(self.d_k, dropout)
        else:
            raise ValueError("Attention type must be 'dot' or 'additive'")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        # 线性变换并分多头
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        if mask is not None:
            # 确保掩码形状正确
            if mask.dim() == 4:  # 解码器掩码 [batch, 1, seq_len, seq_len]
                mask = mask.squeeze(1)  # 变为 [batch, seq_len, seq_len]
            mask = mask.unsqueeze(1)  # 变为 [batch, 1, seq_len, seq_len]
        
        output, attn = self.attention(q, k, v, mask)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous()
        # 计算正确的维度
        output = output.view(batch_size, seq_len_q, -1)
        output = self.W_o(output)
        
        return output, attn
    


# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attention_type='dot', dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, attention_type, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attention_type='dot', dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, attention_type, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, attention_type, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, trg_mask):
        attn_output, _ = self.self_attn(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x

# 编码器
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, d_ff, attention_type='dot', dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, attention_type, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        x = self.dropout(self.pos_encoding(self.embedding(src)))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

# 解码器
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_model, n_layers, n_heads, d_ff, attention_type='dot', dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, attention_type, dropout)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, enc_output, src_mask, trg_mask):
        x = self.dropout(self.pos_encoding(self.embedding(trg)))
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask)
        output = self.fc(x)
        return output

# Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, attention_type='dot', dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, attention_type, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, n_layers, n_heads, d_ff, attention_type, dropout)
    
    def forward(self, src, trg, src_mask, trg_mask):
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_output, src_mask, trg_mask)
        return output

# 生成掩码
def create_src_mask(src):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    return src_mask

def create_trg_mask(trg):
    trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(2)
    trg_len = trg.size(1)
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

# 训练函数
def train(model, train_loader, optimizer, criterion, device, max_batches=None):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for i, (src, trg) in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break
            
        src = src.to(device)
        trg = trg.to(device)
        
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]
        
        src_mask = create_src_mask(src)
        trg_mask = create_trg_mask(trg_input)
        
        optimizer.zero_grad()
        output = model(src, trg_input, src_mask, trg_mask)
        
        loss = criterion(output.reshape(-1, output.size(-1)), trg_output.reshape(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # 打印批次信息
        if i % 10 == 0:
            print('Batch {} of {}, Loss: {:.4f}'.format(i, max_batches if max_batches else len(train_loader), loss.item()))
    
    return epoch_loss / batch_count

# 评估函数
def evaluate(model, test_loader, criterion, device, max_batches=None):
    model.eval()
    epoch_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(test_loader):
            if max_batches is not None and i >= max_batches:
                break
                
            src = src.to(device)
            trg = trg.to(device)
            
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]
            
            src_mask = create_src_mask(src)
            trg_mask = create_trg_mask(trg_input)
            
            output = model(src, trg_input, src_mask, trg_mask)
            
            loss = criterion(output.reshape(-1, output.size(-1)), trg_output.reshape(-1))
            epoch_loss += loss.item()
            batch_count += 1
    
    return epoch_loss / batch_count

# 翻译函数
def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=100):
    model.eval()
    
    # 分词
    tokens = ['<SOS>'] + tokenize_en(sentence) + ['<EOS>']
    src_indices = [src_vocab.stoi.get(token, src_vocab.stoi['<UNK>']) for token in tokens]
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
    
    # 创建掩码
    src_mask = create_src_mask(src_tensor)
    
    # 编码
    enc_output = model.encoder(src_tensor, src_mask)
    
    # 解码
    trg_indices = [trg_vocab.stoi['<SOS>']]
    for i in range(max_len):
        trg_tensor = torch.tensor(trg_indices).unsqueeze(0).to(device)
        trg_mask = create_trg_mask(trg_tensor)
        
        output = model.decoder(trg_tensor, enc_output, src_mask, trg_mask)
        pred_token = output.argmax(dim=-1)[:, -1].item()
        trg_indices.append(pred_token)
        
        if pred_token == trg_vocab.stoi['<EOS>']:
            break
    
    # 转换为单词
    trg_tokens = [trg_vocab.itos[idx] for idx in trg_indices[1:-1]]
    return ' '.join(trg_tokens)

# 配置参数
BATCH_SIZE = 2
D_MODEL = 32  # 减小模型维度以加快训练速度
N_LAYERS = 1  # 减小层数以加快训练速度
N_HEADS = 2  # 减小注意力头数以加快训练速度
D_FF = 64   # 减小前馈网络维度以加快训练速度
DROPOUT = 0.1
EPOCHS = 1   # 减少训练轮数以加快测试速度
MAX_LEN = 10 # 减小最大序列长度以加快训练速度
MAX_TRAIN_BATCHES = 10  # 限制训练批次数量以加快测试速度
MAX_EVAL_BATCHES = 5    # 限制评估批次数量以加快测试速度
MAX_SAMPLES = 100       # 限制样本数量以加快测试速度

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print('Script directory:', SCRIPT_DIR)

# 主函数
def main():
    try:
        # 使用模块级别的配置参数
        
        # 设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: {}".format(device))
        
        # 加载数据集
        print("Loading datasets...")
        print("Loading training data...")
        train_dataset = TranslationDataset(
            'eng-fra_train_data.txt',
            tokenize_en,
            tokenize_fr,
            max_len=MAX_LEN,
            max_samples=MAX_SAMPLES
        )
        print("Train dataset loaded: {} samples".format(len(train_dataset)))
        
        print("Loading testing data...")
        test_dataset = TranslationDataset(
            'eng-fra_test_data.txt',
            tokenize_en,
            tokenize_fr,
            src_vocab=train_dataset.src_vocab,
            trg_vocab=train_dataset.trg_vocab,
            max_len=MAX_LEN,
            max_samples=50
        )
        print("Test dataset loaded: {} samples".format(len(test_dataset)))
        
        print("Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        print("Data loaders created")
        
        print("Source vocabulary size: {}".format(len(train_dataset.src_vocab)))
        print("Target vocabulary size: {}".format(len(train_dataset.trg_vocab)))
        print("Training dataset size: {}".format(len(train_dataset)))
        print("Testing dataset size: {}".format(len(test_dataset)))
        
        # 训练点积注意力模型
        print("\nTraining model with dot product attention...")
        model_dot = Transformer(
            len(train_dataset.src_vocab),
            len(train_dataset.trg_vocab),
            D_MODEL,
            N_LAYERS,
            N_HEADS,
            D_FF,
            attention_type='dot',
            dropout=DROPOUT
        ).to(device)
        
        optimizer_dot = optim.Adam(model_dot.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        best_loss_dot = float('inf')
        for epoch in range(EPOCHS):
            start_time = time.time()
            train_loss = train(model_dot, train_loader, optimizer_dot, criterion, device, MAX_TRAIN_BATCHES)
            test_loss = evaluate(model_dot, test_loader, criterion, device, MAX_EVAL_BATCHES)
            end_time = time.time()
            
            if test_loss < best_loss_dot:
                best_loss_dot = test_loss
                save_path = os.path.join(SCRIPT_DIR, 'transformer_dot.pth')
                print('Saving dot product attention model to:', save_path)
                torch.save(model_dot.state_dict(), save_path)
            
            print("Epoch {}/{}, Train Loss: {}, Test Loss: {}, Time: {}s".format(epoch+1, EPOCHS, str(train_loss)[:6], str(test_loss)[:6], str(end_time - start_time)[:4]))
        
        # 训练加性注意力模型
        print("\nTraining model with additive attention...")
        model_additive = Transformer(
            len(train_dataset.src_vocab),
            len(train_dataset.trg_vocab),
            D_MODEL,
            N_LAYERS,
            N_HEADS,
            D_FF,
            attention_type='additive',
            dropout=DROPOUT
        ).to(device)
        
        optimizer_additive = optim.Adam(model_additive.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        
        best_loss_additive = float('inf')
        for epoch in range(EPOCHS):
            start_time = time.time()
            train_loss = train(model_additive, train_loader, optimizer_additive, criterion, device, MAX_TRAIN_BATCHES)
            test_loss = evaluate(model_additive, test_loader, criterion, device, MAX_EVAL_BATCHES)
            end_time = time.time()
            
            if test_loss < best_loss_additive:
                best_loss_additive = test_loss
                save_path = os.path.join(SCRIPT_DIR, 'transformer_additive.pth')
                print('Saving additive attention model to:', save_path)
                torch.save(model_additive.state_dict(), save_path)
            
            print("Epoch {}/{}, Train Loss: {}, Test Loss: {}, Time: {}s".format(epoch+1, EPOCHS, str(train_loss)[:6], str(test_loss)[:6], str(end_time - start_time)[:4]))
        
        # 加载最佳模型并测试
        print("\nTesting best models...")
        model_dot.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'transformer_dot.pth')))
        model_additive.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'transformer_additive.pth')))
        
        # 评估最终性能
        final_loss_dot = evaluate(model_dot, test_loader, criterion, device, MAX_EVAL_BATCHES)
        final_loss_additive = evaluate(model_additive, test_loader, criterion, device, MAX_EVAL_BATCHES)
        
        print("\nFinal evaluation:")
        print("Dot product attention test loss: {}".format(str(final_loss_dot)[:6]))
        print("Additive attention test loss: {}".format(str(final_loss_additive)[:6]))
        
        if final_loss_dot < final_loss_additive:
            print("Dot product attention performs better!")
        else:
            print("Additive attention performs better!")
    except Exception as e:
        print("Error: " + str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 调用main函数并添加错误处理
    try:
        print("Starting transformer translation task...")
        main()
    except Exception as e:
        print("Error: " + str(e))
        import traceback
        traceback.print_exc()