
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class ConvLeakyRelu2d_3(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self,x):

        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvLeakyRelu2d_1(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self,x):
        out = F.leaky_relu(self.conv(x), negative_slope=0.2)
        return out

class laplacian(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(laplacian, self).__init__()
        sobel_filter = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))

    def forward(self, x):
        laplacian = self.convx(x)
        return laplacian

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x



class Dense(nn.Module):
    def __init__(self,channels):
        super(Dense, self).__init__()
        self.conv1 = ConvLeakyRelu2d_3(channels, channels)
        self.conv2 = ConvLeakyRelu2d_3(2*channels, channels)
        self.conv3 = ConvLeakyRelu2d_3(3*channels, channels)
    def forward(self,x):
        con1_out = self.conv1(x)
        x1 = torch.cat((x, con1_out), dim=1)

        con2_out = self.conv2(x1)
        x2 = torch.cat((x, con1_out), dim=1)
        x2 = torch.cat((x2, con2_out), dim=1)

        con3_out = self.conv3(x2)
        x3 = torch.cat((x, con3_out), dim=1)
        x3 = torch.cat((con1_out, x3), dim=1)
        x3 = torch.cat((con2_out, x3), dim=1)
        return x3

class Block(nn.Module):
    def __init__(self,channels):
        super(Block, self).__init__()
        self.conv1 = ConvLeakyRelu2d_1(channels, channels)
        self.conv2 = ConvLeakyRelu2d_1(2*channels, 2*channels)
        self.conv3 = ConvLeakyRelu2d_3(channels, 2*channels)
        self.conv4 = ConvLeakyRelu2d_3(2*channels, 2*channels)
        self.laplacian = laplacian(channels)
        self.sobel = Sobelxy(channels)
    def forward(self,x):
        con1_tem1 = self.conv3(x)
        con1_tem2 = self.conv4(con1_tem1)
        con1_out = self.conv2(con1_tem2)
        con2_laplacian = self.laplacian(x)

        con2_tem1 = self.conv3(torch.add(x,con2_laplacian))
        con2_tem2 = self.conv4(con2_tem1)
        con2_out = self.conv2(con2_tem2)
        sobel_out = self.sobel(x)
        con3_out = self.conv1(sobel_out)
        con4_out = self.conv1(x)
        x1 = torch.cat((con1_out, con2_out), dim=1)
        x2 = torch.cat((con3_out, con4_out), dim=1)
        con_all_out = torch.cat((x1, x2), dim=1)
        return  con_all_out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_1 = ConvLeakyRelu2d_1(1, 4)
        self.dense = Dense(4)
        self.block = Block(16)
    def forward(self,x):
        con1_out = self.conv_1(x)
        dense_out = self.dense(con1_out)
        block_out = self.block(dense_out)
        return block_out

class ConvTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(ConvTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = ConvLeakyRelu2d_3(192, 96)
        self.conv2 = ConvLeakyRelu2d_3(96, 48)
        self.conv3 = ConvLeakyRelu2d_3(48, 16)
        self.conv4 = ConvTanh2d(16,1)

    def forward(self,x):
        con1_out = self.conv1(x)
        con2_out = self.conv2(con1_out)
        con3_out = self.conv3(con2_out)
        con4_out = self.conv4(con3_out)
        return con4_out

class SIGNet(nn.Module):
    def __init__(self):
        super(SIGNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, image_vis_y,image_ir):
        encoder_ir = self.encoder(image_ir)
        encoder_vis = self.encoder(image_vis_y)
        x = torch.cat((encoder_ir, encoder_vis), dim=1)
        x = self.decoder(x)
        return x








