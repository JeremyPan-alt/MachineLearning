import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np


class WordEmbedding(nn.Module):
    def __init__(self):
        super(WordEmbedding, self).__init__()
        


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # seq1 = torch.randint(0, 8, [5, ])
    # seq2 = torch.randint(0, 9, [8, ])

    # 模拟原序列和目标序列词表中Token的最大索引值
    max_src_index = 10
    max_tgt_index = 20

    # LayerNorm后会把序列补足到最大长度
    max_src_len = 8
    max_tgt_len = 10
    embedding_dim = 512

    # 这里模拟两个句子的翻译情况，第一句子是4个词，第二个句子6个词，翻译成7个词和8个词
    src_len = torch.Tensor([4, 6]).to(torch.int32)
    tgt_len = torch.Tensor([7, 8]).to(torch.int32)

    # 将源、目标的两个句子各自放入一个列表，padding后拼接为一个Tensor
    src_seq_init = [torch.randint(0, max_src_index, (L, )) for L in src_len]
    tgt_seq_init = [torch.randint(0, max_src_index, (L, )) for L in tgt_len]
    src_seq_padded = [F.pad(src_seq_to_pad, (0, max_src_len - len(src_seq_to_pad))) for src_seq_to_pad in src_seq_init]
    tgt_seq_padded = [F.pad(tgt_seq_to_pad, (0, max_tgt_len - len(tgt_seq_to_pad))) for tgt_seq_to_pad in tgt_seq_init]
    src_seq = torch.cat([torch.unsqueeze(src_seq_to_cat, 0) for src_seq_to_cat in src_seq_padded], dim=0)
    tgt_seq = torch.cat([torch.unsqueeze(tgt_seq_to_cat, 0) for tgt_seq_to_cat in tgt_seq_padded], dim=0)

    src_seq.to(device)
    tgt_seq.to(device)

    # 构建Embedding，+1是因为padding引入了0，这个0和索引的0意义不同
    src_seq_embedded = nn.Embedding(max_src_index + 1, embedding_dim).to(device)
    tgt_seq_embedded = nn.Embedding(max_tgt_index + 1, embedding_dim).to(device)

