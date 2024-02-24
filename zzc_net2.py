import torch
import torch.nn as nn
import torch.nn.functional as F
'''
teddy 5
'''

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(628, 480),
            nn.ReLU(),
            nn.Linear(480, 360),
            nn.ReLU(),
            nn.Linear(360, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, 5),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, num_layers=6, heads=8, hidden_size=512):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.EmbeddingBag(300, hidden_size, sparse=True)  # 位置编码
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=heads)
            for _ in range(num_layers)
        ])
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_blocks, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x 的形状为 (batch_size, sequence_length, input_size)
        x = self.embedding(x)
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        pos_enc = self.positional_encoding(positions)

        x += pos_enc

        # Transformer 编码器
        x = x.permute(1, 0, 2)  # 将 batch 维移到前面
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # 恢复原始形状

        # 全连接层进行分类
        x = x.mean(dim=1)  # 对序列维度取平均
        output = self.fc(x)

        return output

# 示例用法
input_size = 9
num_classes = 5
model = TransformerClassifier(input_size, num_classes)

# 随机生成输入数据，形状为 (batch_size, sequence_length, input_size)
input_data = torch.randn(16, 300, input_size)

# 模型前向传播
output = model(input_data)
print(output.shape)


