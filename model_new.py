import torch
import torch.nn as nn
import torch.nn.functional as F
import neighbor_feature


def maxpool_and_cat(x,in_c):
    #(1,N,in_c)
    _,N,C=x.shape
    x=x.permute(0,2,1)
    #(1,in_c,N)
    mpool=nn.Maxpool(in_c)  #in_c为特征数目
    x=mpool(x)
    #(1,in_c,1)
    x=x.permute(0,2,1)
    #(1,1,in_c)
    x=x.expand(1,N,C)
    #(1,N,in_c)
    return x



class Pct_seg(nn.Module):
    def __init__(self,out_c):
        super(Pct_seg, self).__init__()
        self.out_c=out_c
        self.conv1 = nn.Conv1d(628, 256, kernel_size=1, bias=False)
        #self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        #self.conv3 = nn.Conv1d(128,64 , kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(256,128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        #self.bn2 = nn.BatchNorm1d(64)
        #self.bn3=nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.pt_last = Point_Transformer_Last(channels=128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(756, 512, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(512),
                                       nn.LeakyReLU(negative_slope=0),
                                       nn.Conv1d(512, 128, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(128),
                                       nn.LeakyReLU(negative_slope=0),

                                       nn.Conv1d(128, out_channels=self.out_c, kernel_size=1, bias=False),

                                       )

        self.seg = nn.Sequential(nn.Conv1d(8, 8, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(8),
                                 nn.LeakyReLU(negative_slope=0.2),

                                 nn.Conv1d(8, out_channels=self.out_c, kernel_size=1, bias=False),

                                 )



    def forward(self, x):
        #--------------------- Input Embedding --------------------
        # Neighbor Embedding: LBR --> LBR --> LBR --> SG
        #x:(B,N,628)
        #global_feature=maxpool_and_cat(x,628)
        x = x.permute(0, 2, 1)
        # x (B,628,N)
        x = F.relu(self.bn1(self.conv1(x)))
        #(1,256,N)
        x = F.relu(self.bn4(self.conv4(x)))
        #(1,128,n)

        self.before_transformer=x
        #----------------- Self Attention -------------------------------------


        #(1,128,N)
        #print(x.shape)
        x = self.pt_last(x)

        #x = x.permute(0, 2, 1)
        #x=x+self.before_transformer
        #(1,128, N)
        #x = torch.cat([x,self.before_transformer],dim=1) # 32
        x=x.permute(0,2,1)
        #(1,N,128)
        x=torch.cat([x,global_feature],dim=2) #128+628=756
        #1,32,n

        x = self.conv_fuse(x)                # in:756, out:c

        #x=self.seg(self.face_feature)  #in 8 out c
        # Point Feature --> Global Feature --> LBRD --> LBR --> Linear --> Predict label
        #(1,C,N)
        x = x.permute(0, 2, 1)
        #(1,N,C)
        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()

        self.conv1 = nn.Conv1d(channels+3, channels, kernel_size=1, bias=False)
        #self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.v,self.f=read_off(args.off_path)
        self.position_feature=position_encoding(self.v,self.f) #N,3
        self.bn1 = nn.BatchNorm1d(channels)
        #self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        #self.sa2 = SA_Layer(channels)
        #self.sa3 = SA_Layer(channels)
        #self.sa4 = SA_Layer(channels)

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        self.position=torch.unsqueeze(self.position,0).permute(0,2,1)
        #1,3,N
        x=torch.cat([x,self.position],dim=1)
        #(1,3+F,N)

        x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        #x2 = self.sa2(x1)
        #x3 = self.sa3(x2)
        #x4 = self.sa4(x3)
        #x = torch.cat([x1,x], dim=1)

        return x1

    def read_off(off_path):
        f = open(data_dir, 'r')
        f.readline()
        num_v, num_f, num_edges = map(int, f.readline().split())
        v_data = []
        for view_id in range(num_v):
            v_data.append(list(map(float, f.readline().split())))
        v_data = np.array(v_data)
        f_data = []
        for face_id in range(num_f):
            f_data.append(list(map(int, f.readline().split()[1:])))
        f_data = np.array(f_data)
        f.close()

        return v_data,f_data

    def position_encoding(v,f):
        l=len(f)
        position=[]
        for i in range(1):
            a,b,c=f[i]

            center=v[0].copy()
            center[0]=(v[a-1][0]+v[b-1][0]+v[c-1][0])/3
            center[1]=(v[a-1][1]+v[b-1][1]+v[c-1][1])/3
            center[2]=(v[a-1][2]+v[b-1][2]+v[c-1][2])/3
            position.append(center)
        position=np.array(position)
        #(N,3)
        return position


# self attention layer
class SA_Layer(nn.Module):
    def __init__(self, channels):
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
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x