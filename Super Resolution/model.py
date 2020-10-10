# SRGAN model
import math
import torch
import torch.nn as nn

# Residual block

class ResidualBlock(nn.Module):
    '''
    Convolution + Batch Normalization + PRelu -> Convolution + Batch Normalization -> skip connection
    Inp
    '''
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        residue = self.conv1(x)
        residue = self.bn1(residue)
        residue = self.prelu(residue)
        residue = self.conv2(residue)
        residue = self.bn2(residue)
        return x + residue

class UpsampleBlock(nn.Module):
    '''
    Upsample block - upgrades the input by a upscale factor. Resulting the height and width of the input to be multiplied by the upscaling factor.
    Convolution -> PixelShuffle -> Prelu
    '''
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=up_scale)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    '''
    Generator block - Generate upscaled images.
    Flow: 9x9 conv -> 5 residual blocks -> Upsample -> Skip connection
    '''
    def __init__(self, scale_factor):
        upsample_block_number = int(math.log(scale_factor, 2))
        super(Generator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64)
        )
        
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_number)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
    
    def forward(self, x):        
        block1 = self.block1(x) # 64 channels
        gen_block = self.block2(block1)
        gen_block = self.block3(gen_block)
        gen_block = self.block4(gen_block)
        gen_block = self.block5(gen_block)
        gen_block = self.block6(gen_block)
        gen_block = self.block7(gen_block) # 64 channels
        x = self.block8(gen_block + block1)
        return (torch.tanh(x) + 1) /2


class ConvolutionalBlock(nn.Module):
    '''
    Convolutional block - sequence of convolution + BatchNorm + LeakyRelu
    conv(x) with stride 2 -> BN -> LeakyRelu -> conv(2x) -> BN -> LeakyRelu
    '''
    def __init__(self, channels):
        super(ConvolutionalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.leakyrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels= 2*channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(2*channels)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)
        return x



class Discriminator(nn.Module):
    '''
    Discriminator block - classify if the image is fake or not.
    '''
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2)
        )
        self.block2 = ConvolutionalBlock(64)
        self.block3 = ConvolutionalBlock(128)
        self.block4 = ConvolutionalBlock(256)

        self.block5 = nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2)
        )

        self.block6 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return torch.sigmoid(x.view(batch_size))