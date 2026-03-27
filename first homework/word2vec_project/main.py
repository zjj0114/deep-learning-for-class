###小组成员为：
###1. 章建建
###2. 潘义明
###3. 吴辰扬
###4. 吕志远

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 读取和预处理数据
def read_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    # 简单分词
    words = re.findall(r'\b\w+\b', text)
    return words

# 构建词汇表
def build_vocabulary(words):
    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(set(words)):
        word_to_idx[word] = i
        idx_to_word[i] = word
    return word_to_idx, idx_to_word

# 构建共现矩阵
def build_cooccurrence_matrix(words, word_to_idx, window_size=5):
    vocab_size = len(word_to_idx)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for i, word in enumerate(words):
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                matrix[word_to_idx[word]][word_to_idx[words[j]]] += 1
    
    return matrix

# 训练词向量（使用SVD）
def train_word_vectors(cooccurrence_matrix, embedding_dim=100):
    # 计算PMI矩阵
    total = np.sum(cooccurrence_matrix)
    row_sums = np.sum(cooccurrence_matrix, axis=1, keepdims=True)
    col_sums = np.sum(cooccurrence_matrix, axis=0, keepdims=True)
    pmi = np.log((cooccurrence_matrix * total) / (row_sums * col_sums + 1e-10))
    pmi = np.maximum(pmi, 0)  # 只保留正PMI值
    
    # 使用SVD降维
    u, s, vh = np.linalg.svd(pmi)
    word_vectors = u[:, :embedding_dim] * np.sqrt(s[:embedding_dim])
    
    return word_vectors

# 可视化词向量
def visualize_word_vectors(word_vectors, idx_to_word, num_words=10):
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建output文件夹
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用PCA降维到2维
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(word_vectors[:num_words])
    
    # 绘制词向量分布
    plt.figure(figsize=(10, 8))
    for i in range(num_words):
        word = idx_to_word[i]
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title('Word Vector Distribution (PCA)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    
    # 保存图像到output文件夹
    image_path = os.path.join(output_dir, 'word_vector_distribution.png')
    plt.savefig(image_path)
    plt.show()
    
    print(f"词向量分布图像已保存为: {image_path}")

if __name__ == "__main__":
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据和模型路径
    data_path = os.path.join(current_dir, 'data', 'sample_text.txt')
    model_path = os.path.join(current_dir, 'models', 'word_vectors.npy')
    
    # 读取和预处理数据
    print("正在读取训练数据...")
    words = read_and_preprocess(data_path)
    
    # 构建词汇表
    print("正在构建词汇表...")
    word_to_idx, idx_to_word = build_vocabulary(words)
    
    # 构建共现矩阵
    print("正在构建共现矩阵...")
    cooccurrence_matrix = build_cooccurrence_matrix(words, word_to_idx)
    
    # 训练词向量
    print("正在训练词向量...")
    word_vectors = train_word_vectors(cooccurrence_matrix)
    
    # 保存模型
    print(f"正在保存模型到: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    np.save(model_path, word_vectors)
    # 使用绝对路径保存其他模型文件
    word_to_idx_path = os.path.join(current_dir, 'models', 'word_to_idx.npy')
    idx_to_word_path = os.path.join(current_dir, 'models', 'idx_to_word.npy')
    np.save(word_to_idx_path, word_to_idx)
    np.save(idx_to_word_path, idx_to_word)
    
    # 可视化词向量
    print("正在可视化词向量...")
    visualize_word_vectors(word_vectors, idx_to_word, num_words=10)
    
    print("项目执行完成！")
