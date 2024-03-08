import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import my_dataset


class MLP(nn.Module):
    def __init__(self, args, out_c):
        super(MLP, self).__init__()
        self.args = args                         # 参数
        self.out_c = args.out_c                  # 输出类别数
        self.model = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, self.out_c),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x.to(self.device)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size              # 512
        self.heads = heads                        # 8头，切成8个部分
        self.head_dim = embed_size // heads       # 512/8=64   8头自注意力，每个头维度都是         64*3

        # assert (self.head_dim * heads == embed_size), "Embed size needs  to  be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False).to('cuda')    # 64*64
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False).to('cuda')
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False).to('cuda')  # 都是64*64的全连接,就是矩阵
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size).to('cuda')          # 输出是512*512

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # 训练样本的数量，batch数    16
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # 都是300

        # Split embedding into self.heads pieces  把kqv都切成八头  values为(4, 300, 1536)
        values = values.reshape(N, value_len, self.heads, self.head_dim)  # 512变成8头64维
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # 过一遍神经网络，就是乘了个参数矩阵Wv
        keys = self.keys(keys)
        queries = self.queries(queries)
        # attention公式里面QK^T就是这个  矩阵乘法
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (16, 8, 300, 300) 两个序列，8个头，q和k
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:  # energy 就变成了下三角矩阵
            energy = energy.masked_fill(mask == 0, float("-1e20"))   # 这里就是掩码，把energy遮起来
            # Fills elements of self tensor with value where mask is True

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (16, 8, 300, 300)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(  # 把attention*value 矩阵乘法
            N, query_len, self.heads * self.head_dim
        )  # (16, 300, 8, 64)->(16, 300, 512)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out.to('cuda')   # 输入输出都是(16, 300, 512) 但是经过了注意力的计算


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size).to('cuda')
        self.norm2 = nn.LayerNorm(embed_size).to('cuda')
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        ).to('cuda')
        self.dropout = nn.Dropout(dropout).to('cuda')

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask).to('cuda')  # (2, 9, 256)

        x = self.dropout(self.norm1(attention + query))  # 图上的add和norm， add就是残差
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # (2, 9, 256)  有个残差结构
        return out.to('cuda')


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,    # 10  源词向量维度为10
                 embed_size,        # 512
                 num_layers,
                 heads,             # 8
                 device,
                 forward_expansion,   # 4
                 dropout,           # 0
                 max_length,
                 feature_num
                 ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # self.word_embedding = nn.Embedding(src_vocab_size, embed_size)   # (10, 256)  词汇表大小，embedding大小
        self.word_embedding = nn.Linear(feature_num, embed_size).to(self.device)   # 编码器就是个mlp，学出来的
        self.position_embedding = nn.Embedding(max_length, embed_size).to(self.device)   # (300, 512)  位置编码

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
        N, seq_lengh, fea_len = x.shape   # (16, 300, 5)
        # 这就是1234累加的位置编码，后面要改
        positions = torch.arange(0, seq_lengh).expand(N, seq_lengh).to(self.device)   # 16个0到299的列表 (16,300)
        # pe = torch.zeros(seq_lengh, 256*3)
        # position = torch.arange(0, seq_lengh).unsqueeze(1).float()
        # div_term = torch.exp(torch.arange(0., 256*3, 2) * -(math.log(10000.0) / 256*3))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.expand(N, seq_lengh, 256*3).to(self.device)
        emb = self.word_embedding(x).reshape(N, 300, -1)   # (16, 300, 512)
        pos = self.position_embedding(positions)   # 位置编码(16, 300, 512)
        # emb = emb.to(self.device)
        # pos = pos.to(self.device)
        # out = self.dropout(self.word_embedding(x).reshape(N, 300, -1).to(self.device) + self.position_embedding(positions).to(self.device))  # 位置编码加上去
        out = self.dropout(emb + pos)  # 位置编码加上去
        # out = self.dropout(self.word_embedding(x).reshape(4, 300, -1) + pe)  # 位置编码加上去

        for layer in self.layers:   # 6层 transformer
            out = layer(out, out, out, mask)   # (16,300,512)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads).to('cuda')
        self.norm = nn.LayerNorm(embed_size).to('cuda')
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion).to('cuda')
        self.dropout = nn.Dropout(dropout).to('cuda')

    def forward(self, x, value, key, src_mask, trg_mask):  # 不同语言之间的关联
        attention = self.attention(x, x, x, trg_mask)      # 算标签语言自己的
        query = self.dropout(self.norm(attention + x))     # 解码器自己提供q，resnet回路
        out = self.transformer_block(value, key, query, src_mask)  # k和v是编码器对面片的编码结果，q是自己
        return out.to('cuda')


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
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size).to('cuda')
        self.position_embedding = nn.Embedding(max_length, embed_size).to('cuda')

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]  # 叠6层
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size).to('cuda')   # MLP 把维度降下来，输出结果
        self.dropout = nn.Dropout(dropout).to('cuda')

    def forward(self, x, enc_out, src_mask, trg_mask):  # 标签，编码后的词汇，两个mask
        N, seq_length = x.shape   # 16, 300   # TODO：标签需要编码到512维吗
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  # (16, 300)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))  # (2, 7, 256)  先词嵌入再加上位置编码再随即失活

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)   # 两种语言之间计算，x是输出语言的，enc_out是输入语言的

        out = self.fc_out(x)
        return out.to(self.device)


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.device = args.device
        # self.src_vocab_size = args.src_vocab_size
        # self.trg_vocab_size = args.trg_vocab_size
        # self.src_pad_idx = args.src_pad_idx
        # self.trg_pad_idx = args.trg_pad_idx
        self.encoder = Encoder(
            src_vocab_size=self.args.src_vocab_size,
            embed_size=self.args.embed_size,
            num_layers=self.args.num_layers,
            heads=self.args.heads,
            device=self.device,
            forward_expansion=self.args.forward_expansion,
            dropout=self.args.dropout,
            max_length=self.args.seq_len,
            feature_num=self.args.feature_num
        )

        self.decoder = Decoder(
            args.trg_vocab_size,
            args.embed_size,
            args.num_layers,
            args.heads,
            args.forward_expansion,
            args.dropout,
            self.device,
            args.max_length
        )
        self.src_pad_idx = args.src_pad_idx
        self.trg_pad_idx = args.trg_pad_idx
        # self.device = device

    def make_src_mask(self, src):
        """生成输入序列的掩码,0的地方就设置为false其他为true"""
        src = src.sum(dim=-1)  # (16, 300, 5)->(16, 300)   最后五个特征合起来，只要形状，生产掩码
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (16, 1, 1, 300)
        # (N, 1, 1, src_len)  我这里全是true因为我的长度是固定的
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape   # 批量大小，序列长度  (16, 300)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)  # (16, 1, 300, 300) 全下三角矩阵
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  # (16, 1, 1, 300)
        trg_mask = self.make_trg_mask(trg)  # (16, 1, 300, 300)
        enc_src = self.encoder(src, src_mask)  # (16, 300, 512)  从词向量变成理解向量
        out = self.decoder(trg, enc_src, src_mask, trg_mask)  # (16, 300, 5)
        return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mesh Transformer')  # 使用命令行参数
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--full_test', type=bool, default=False)  # 是否把面片索引作为特征加上去,训练模式
    parser.add_argument('--n_walks_per_model', type=int, default=4)  # 4*4=16，每次16条序列

    # 关于网络的参数
    parser.add_argument('--src_vocab_size', type=int, default=1000)  # 面片词汇大小
    parser.add_argument('--trg_vocab_size', type=int, default=5)     # 标签词汇大小
    parser.add_argument('--src_pad_idx', type=int, default=0)        #
    parser.add_argument('--trg_pad_idx', type=int, default=0)        #

    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--forward_expansion', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=300)

    parser.add_argument('--seq_len', type=int, default=300)  # 做消融实验，不一定是300，我的模型挺大的
    parser.add_argument('--feature_num', type=int, default=5, help='Number of features')  # 消融实验

    args = parser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        args.device = "cuda"
        print('Using GPU')
    else:
        args.device = "cpu"
        print('Using CPU')

    path = 'datasets_processed/psb/psb_ant'
    dataset = my_dataset.PSBDataset(args, path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # src_pad_idx = 0   # 不定长序列中用0填充
    # trg_pad_idx = 0
    # src_vocab_size = 1024        # 10  输入序列的词汇表大小
    # trg_vocab_size = 5           # 10  就五个类别，五个词够了
    model = Transformer(args).to(args.device)
    for data, label in dataloader:
        data = torch.tensor(data, dtype=torch.float32)   # (4, 1200, 5)
        label = label.to(dtype=torch.int64)              # (4, 1200)
        data, label = data.to(args.device), label.to(args.device)
        data = data.view((-1, 300, 5))   # (16, 300, 5)
        label = label.view(-1, 300)      # (16, 300)

        out = model(data, label)
        print(out.shape)   # (16, 300, 5)
        # print(out)








