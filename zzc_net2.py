import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size          # 256  把这256的长度，切成8个部分               512*3
        self.heads = heads                    # 8
        self.head_dim = embed_size // heads   # 256/8=32 8头自注意力，每个头维度都是32         64*3

        assert (self.head_dim * heads == embed_size), "Embed size needs  to  be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)  # 都是32x32的全连接,就是矩阵
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)          # 输出是256x256

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # 训练样本的数量，batch数
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # 每个单词的词向量维度 9

        # Split embedding into self.heads pieces  把kqv都切成八头
        values = values.reshape(N, value_len, self.heads, self.head_dim)  # (2, 9, 256)->(2, 9, 8, 32)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # 过一遍神经网络，就是乘了个参数矩阵Wv
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # 张量收缩 (2, 8, 9, 9) 两个序列，8个头，q和k
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # Fills elements of self tensor with value where mask is True

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (2, 8, 9, 9)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(  # 把attention和value拼接
            N, query_len, self.heads * self.head_dim
        )  # (2, 9, 8, 32)->(2, 9, 256)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out   # 输入输出都是(2, 9, 256) 但是经过了注意力的计算


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)  # (2, 9, 256)

        x = self.dropout(self.norm1(attention + query))  # 图上的add和norm， add就是残差
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # (2, 9, 256)  有个残差结构
        return out


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,    # 10  源词向量维度为10
                 embed_size,        # 256  encoder后的维度是 256       512
                 num_layers,
                 heads,             # 8
                 device,
                 forward_expansion,   # 4
                 dropout,           # 0
                 max_length
                 ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)   # (10, 256)  词汇表大小，embedding大小
        self.position_embedding = nn.Embedding(max_length, embed_size)   # (100, 256)  位置编码

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)]   # 循环6次
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_lengh, fea_len = x.shape   # 批次，序列长度
        # 这就是1234累加的位置编码，后面要改
        # positions = torch.arange(0, seq_lengh).expand(N, seq_lengh).to(self.device)
        pe = torch.zeros(seq_lengh, 512*3)
        position = torch.arange(0, seq_lengh).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 512*3, 2) * -(math.log(10000.0) / 512*3))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.expand(N, seq_lengh, 512*3).to(self.device)
        # out = self.dropout(self.word_embedding(x).reshape(4, 300, -1) + self.position_embedding(positions))  # 位置编码加上去
        out = self.dropout(self.word_embedding(x).reshape(4, 300, -1) + pe  # 位置编码加上去

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):  # 不同语言之间的关联吗
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))   # 解码器自己提供q
        out = self.transformer_block(value, key, query, src_mask)  # k和v是编码器提供的
        return out


class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]  # 叠6层
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)   # 从256回到10
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):  # (2, 7)(2, 9, 256)(2, 1, 1, 9)(2, 1, 7, 7)
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  # (2, 7)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))  # (2, 7, 256)  先词嵌入再加上位置编码再随即失活

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)   # 两种语言之间计算，x是输出语言的，enc_out是输入语言的

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,   # 10 源词汇表大小
                 trg_vocab_size,   # 10 目标词汇表大小
                 src_pad_idx,      # 0
                 trg_pad_idx,      # 0
                 embed_size=512*3,
                 num_layers=6,
                 forward_expansion=4,  # 处理输入序列长度不一致，输入序列的末尾添加标记，太长截断，太短填充
                 heads=8,
                 dropout=0,  # 神经元失活，正则化，防过拟合
                 device="cuda",
                 max_length=300    # 100 一个句子最长也就这么长了，用这个设置位置编码
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """生成输入序列的掩码,0的地方就设置为false其他为true"""
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (2, 9)->(2, 1, 1, 9)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape   # 批量大小，序列长度
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)  # 二维下三角掩码
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  # (2, 1, 9, 9)
        trg_mask = self.make_trg_mask(trg)  # (2, 1, 7, 7)
        enc_src = self.encoder(src, src_mask)  # (2, 9, 256)  从词向量变成理解向量
        out = self.decoder(trg, enc_src, src_mask, trg_mask)  # (2, 7, 10)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)   # (2, 9) 输入语言
    x = torch.randint(1, 100, size=(4, 300, 3)).to(device)   # (4, 300, 3)
    # trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)  # (2, 8)  标签，另一种语言
    trg = torch.randint(1, 6, size=(4, 300)).to(device)  # (4, 300)

    src_pad_idx = 0   # 不定长序列中用0填充
    trg_pad_idx = 0
    src_vocab_size = 1024        # 10  输入序列的词汇表大小
    trg_vocab_size = 5           # 10  就五个类别，五个词够了
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    # out = model(x, trg[:, :-1])  # (2, 7, 10)  最后一个位置通常作为预测的目标
    out = model(x, trg)  # (4, 300, 5)  最后一个位置通常作为预测的目标
    print(out.shape)
    print(out)



