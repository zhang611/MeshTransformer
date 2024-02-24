import torch
import torch.nn as nn
import torch.nn.functional as F


# import neighbor_feature


class Pct_seg(nn.Module):
    def __init__(self, args, out_c):
        super(Pct_seg, self).__init__()
        self.args = args                         # 参数
        self.out_c = args.out_c                  # 输出类别数
        self.at_layer = args.attention_layers    # 默认0
        self.offset = args.offset
        self.param = args.input_layer    # (628, 1024, 256, 16)
        self.conv1 = nn.Conv1d(self.param[0], self.param[1], kernel_size=1, bias=False)  # 就是MLP
        self.conv2 = nn.Conv1d(self.param[1], self.param[2], kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.param[2], self.param[3], kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(self.param[1])
        self.bn2 = nn.BatchNorm1d(self.param[2])
        self.bn3 = nn.BatchNorm1d(self.param[3])
        # self.bn4 = nn.BatchNorm1d(128)

        self.dp1 = nn.Dropout(p=args.dropout)

        self.pt_last = Point_Transformer_Last(channels=self.param[3], layers=self.at_layer, offset=self.offset)

        if self.at_layer >= 2:   # 估计这个是多头自注意力机制
            self.channel = self.param[3] * self.at_layer
        else:
            self.channel = self.param[3]   # 单头就是16
        self.seg = nn.Sequential(   # 分割网络也是 MLP
            nn.Conv1d(self.channel, self.channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channel // 2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv1d(self.channel // 2, self.channel // 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channel // 4),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv1d(self.channel // 4, out_channels=self.out_c, kernel_size=1, bias=False),

        )

        self.conv_fuse = nn.Sequential(nn.Conv1d(self.channel, self.channel // 2, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(self.channel // 2),
                                       nn.LeakyReLU(negative_slope=0.2),
                                       )

    def forward(self, x):
        # --------------------- Input Embedding --------------------
        # Neighbor Embedding: LBR --> LBR --> LBR --> SG
        # x:(B,N,628)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # x (B,628,N)
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))  # 首先三层MLP
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = x.permute(0, 2, 1)
        # (B,N,64)
        # 连接邻接特征
        # x = neighbor_feature.cat_neighbor_features(x, idx_in_all)
        # x = torch.unsqueeze(x, dim=0)
        # (1,n,64*4=256)
        # x = x.cuda()

        # (B,N,256)
        # x = x.permute(0, 2, 1)
        # (1,256,n)
        # x = F.relu(self.bn4(self.conv4(x)))
        # (1,64,n)

        self.before_transformer = x   #(1, 16, 21758)
        # ----------------- Self Attention -------------------------------------

        # (1,64,N)
        # print(x.shape)

        if self.at_layer >= 1:
            x = self.pt_last(x)

        # -------- Offset Attention ----------
        # x = torch.cat([self.before_transformer,x], dim=1)
        # ------------------------------------

        # x = x.permute(0, 2, 1)
        # x=x+self.before_transformer
        # (1,16, N)
        face_feature = self.conv_fuse(x)  # in:128, out:64
        x = face_feature.permute(0, 2, 1)  # (1, 21758, 8)
        global_feature = F.adaptive_max_pool1d(face_feature, 1)  # (1, 8, 1)
        global_feature_rep = global_feature.repeat(1, 1, face_feature.shape[2])  # (1, 8, 21758)
        # global_feature_rep = global_feature_rep.permute(0,2,1)
        # print(face_feature.shape)
        # print(global_feature.shape)
        # print(global_feature_rep.shape)
        x = torch.cat([face_feature, global_feature_rep], dim=1)  # 连接全局特征和局部特征(1, 16, 21758)
        # print(x.shape)
        x = self.seg(x)  # in 128*4 out c  (1, 5, 21758)
        # x = self.seg(self.face_feature)  # in 32 out c
        # Point Feature --> Global Feature --> LBRD --> LBR --> Linear --> Predict label
        # (1,C,N)
        x = x.permute(0, 2, 1)  # (1, 21758, 5)
        # (1,N,C)
        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256, layers=1, offset=True):
        super(Point_Transformer_Last, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels, offset)
        self.sa2 = SA_Layer(channels, offset)
        self.sa3 = SA_Layer(channels, offset)
        self.sa4 = SA_Layer(channels, offset)
        self.sa5 = SA_Layer(channels, offset)
        self.sa6 = SA_Layer(channels, offset)
        self.sa7 = SA_Layer(channels, offset)

        self.layers = layers

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        layers = self.layers
        if layers >= 1:
            x1 = self.sa1(x)
            x = x1
        if layers >= 2:
            x2 = self.sa2(x1)
            x = torch.cat([x1, x2], dim=1)
        if layers >= 3:
            x3 = self.sa2(x2)
            x = torch.cat([x1, x2, x3], dim=1)
        if layers >= 4:
            x4 = self.sa2(x3)
            x = torch.cat([x1, x2, x3, x4], dim=1)
        if layers >= 5:
            x5 = self.sa2(x4)
            x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        if layers >= 6:
            x6 = self.sa2(x5)
            x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        if layers >= 7:
            x7 = self.sa2(x6)
            x = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=1)
        # x2 =self.sa2(x1)
        # x3 = self.sa3(x2)
        # x4 = self.sa4(x3)
        # x = x1
        # x = torch.cat([x1,x2], dim=1)
        return x


# self attention layer
class SA_Layer(nn.Module):
    def __init__(self, channels, offset):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 2, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 2, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.offset = offset

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        if self.offset:
            x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  # off-set attention
            x = x + x_r
        else:
            x = self.act(self.after_norm(self.trans_conv(x_r)))

        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  #off-set attention
        # x = x + x_r
        return x
