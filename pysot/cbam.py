import torch
import torch.nn as nn

def conv3x3(in_planes,out_planes,stride=1):
    """3x3 convolutions with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

class ChannelAttentoion(nn.Module):
    def __init__(self,in_planes,ratio=4):
        super(ChannelAttentoion,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes,in_planes//ratio,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio,in_planes,1,bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpitialAttention(nn.Module):
    def __init__(self,kernel_size = 3):
        super(SpitialAttention,self).__init__()
        assert kernel_size in (3,7),"kernel size must be 3 or 7"
        padding = 3 if kernel_size ==7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size,padding=padding,bias= False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgout = torch.mean(x,dim=1,keepdim=True)
        maxout,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avgout,maxout],dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


# class CBAM(nn.Module):
#     def __init__(self,planes = 256):
#         self.ca = ChannelAttentoion()
#         self.sa = SpitialAttention()
#
#     def forward(self,x):
#         x = self.ca(x) * x
#         x = self.sa(x) * x
#         return x;
