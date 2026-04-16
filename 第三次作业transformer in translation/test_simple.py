# 简单测试脚本
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from collections import Counter
import re
import os

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 简单分词函数
def tokenize_en(text):
    text = text.lower()
    text = re.sub(r'[\.,!\?"\'\(\)\[\]]', ' ', text)
    tokens = text.split()
    return tokens

def tokenize_fr(text):
    text = text.lower()
    text = re.sub(r'[\.,!\?"\'\(\)\[\]]', ' ', text)
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
    def __init__(self, file_path, src_tokenizer, trg_tokenizer, src_vocab=None, trg_vocab=None, max_len=100, max_samples=100):
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
                    if i >= max_samples:
                        break
                    
                    line = line.strip()
                    if line:
                        try:
                            if '\t' in line:
                                src, trg = line.split('\t', 1)
                            elif '→' in line:
                                src, trg = line.split('→', 1)
                            else:
                                continue
                            self.src_sentences.append(src.strip())
                            self.trg_sentences.append(trg.strip())
                        except Exception as e:
                            if i < 5:
                                print('Error processing line', i, ':', e)
            
            print('Number of valid sentences:', len(self.src_sentences))
            
            # 检查是否有有效的句子
            if len(self.src_sentences) == 0:
                raise ValueError('No valid sentences found in the dataset. Please check the file format.')
            
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
        except Exception as e:
            print('Error reading file:', e)
            import traceback
            traceback.print_exc()
            # 如果发生异常，确保数据集不为空
            if len(self.src_sentences) == 0:
                raise ValueError('Failed to load dataset. Please check the file path and format.')
    
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
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
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
        q = self.W_q(q)
        k = self.W_k(k)
        
        q = q.unsqueeze(3)
        k = k.unsqueeze(2)
        
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_type='dot', dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
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
        
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            mask = mask.squeeze(1)
        
        output, attn = self.attention(q, k, v, mask)
        
        output = output.transpose(1, 2).contiguous()
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

# 主函数
def main():
    try:
        # 配置参数
        BATCH_SIZE = 2
        D_MODEL = 32
        N_LAYERS = 1
        N_HEADS = 2
        D_FF = 64
        DROPOUT = 0.1
        EPOCHS = 1
        MAX_LEN = 10
        MAX_TRAIN_BATCHES = 10
        MAX_EVAL_BATCHES = 5
        
        # 设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        
        # 加载数据集
        print('Loading datasets...')
        print('Loading training data...')
        train_dataset = TranslationDataset(
            'eng-fra_train_data.txt',
            tokenize_en,
            tokenize_fr,
            max_len=MAX_LEN,
            max_samples=100
        )
        print('Train dataset loaded:', len(train_dataset), 'samples')
        
        if len(train_dataset) == 0:
            print('No training data loaded!')
            return
        
        print('Loading testing data...')
        test_dataset = TranslationDataset(
            'eng-fra_test_data.txt',
            tokenize_en,
            tokenize_fr,
            src_vocab=train_dataset.src_vocab,
            trg_vocab=train_dataset.trg_vocab,
            max_len=MAX_LEN,
            max_samples=50
        )
        print('Test dataset loaded:', len(test_dataset), 'samples')
        
        if len(test_dataset) == 0:
            print('No testing data loaded!')
            return
        
        print('Creating data loaders...')
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        print('Data loaders created')
        
        print('Source vocabulary size:', len(train_dataset.src_vocab))
        print('Target vocabulary size:', len(train_dataset.trg_vocab))
        
        # 训练点积注意力模型
        print('\nTraining model with dot product attention...')
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
            train_loss = train(model_dot, train_loader, optimizer_dot, criterion, device, MAX_TRAIN_BATCHES)
            test_loss = evaluate(model_dot, test_loader, criterion, device, MAX_EVAL_BATCHES)
            
            if test_loss < best_loss_dot:
                best_loss_dot = test_loss
                torch.save(model_dot.state_dict(), os.path.join(SCRIPT_DIR, 'transformer_dot.pth'))
            
            print('Epoch', epoch+1, '/', EPOCHS, 'Train Loss:', train_loss, 'Test Loss:', test_loss)
        
        # 训练加性注意力模型
        print('\nTraining model with additive attention...')
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
            train_loss = train(model_additive, train_loader, optimizer_additive, criterion, device, MAX_TRAIN_BATCHES)
            test_loss = evaluate(model_additive, test_loader, criterion, device, MAX_EVAL_BATCHES)
            
            if test_loss < best_loss_additive:
                best_loss_additive = test_loss
                torch.save(model_additive.state_dict(), os.path.join(SCRIPT_DIR, 'transformer_additive.pth'))
            
            print('Epoch', epoch+1, '/', EPOCHS, 'Train Loss:', train_loss, 'Test Loss:', test_loss)
        
        # 加载最佳模型并测试
        print('\nTesting best models...')
        model_dot.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'transformer_dot.pth')))
        model_additive.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'transformer_additive.pth')))
        
        # 评估最终性能
        final_loss_dot = evaluate(model_dot, test_loader, criterion, device, MAX_EVAL_BATCHES)
        final_loss_additive = evaluate(model_additive, test_loader, criterion, device, MAX_EVAL_BATCHES)
        
        print('\nFinal evaluation:')
        print('Dot product attention test loss:', final_loss_dot)
        print('Additive attention test loss:', final_loss_additive)
        
        if final_loss_dot < final_loss_additive:
            print('Dot product attention performs better!')
        else:
            print('Additive attention performs better!')
    except Exception as e:
        print('Error:', e)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print('Starting transformer translation task...')
    main()
