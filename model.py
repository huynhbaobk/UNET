import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            #Second Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #Second Conv
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512 ]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.sigmoid = nn.Sigmoid()

        ### Encoder part for UNET
        for feature in features:
             self.downs.append(DoubleConv(in_channels, feature))
             in_channels = feature

        ### Decoder part for UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(2*feature, feature, kernel_size=2, stride=2))
            self.ups.append(
                DoubleConv(2*feature, feature)) 
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) 
        # self.softmax = nn.Softmax2d()
        
    def forward(self, x):
        H, W = x.shape[2:]
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)
        print(x.shape)
        # x = self.softmax(x)

        # x = x.view(-1, H*W*4)
        # x = x.unsqueeze(1).unsqueeze(-1)

        return x

def test():
    X = torch.randn(3, 3, 224, 224)
    model = UNET(in_channels=3, out_channels=4)
    y_preds = model(X)
    print(X.shape)
    print(y_preds.shape)

if __name__ == '__main__':
    test()