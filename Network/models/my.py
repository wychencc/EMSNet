import torch
from torch import nn, optim
import torch.nn.functional as F
from functools import reduce


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
class SAGate(nn.Module):
    def __init__(self, channels, out_ch, reduction=16):
        super(SAGate, self).__init__()
        self.channels = channels

        self.fusion1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

        self.softmax = nn.Softmax(dim=1)

        self.fusion2 = nn.Conv2d(channels * 2, out_ch, kernel_size=1)

    def forward(self, sar, opt):
        b, c, h, w = sar.size()
        output = [sar, opt]

        fea_U = self.fusion1(torch.cat([sar, opt], dim=1))
        fea_s = self.avg_pool(fea_U) + self.max_pool(fea_U)
        attention_vector = self.gate(fea_s)
        attention_vector = attention_vector.reshape(b, 2, self.channels, -1)
        attention_vector = self.softmax(attention_vector)
        attention_vector = list(attention_vector.chunk(2, dim=1))
        attention_vector = list(map(lambda x: x.reshape(b, self.channels, 1, 1), attention_vector))
        V = list(map(lambda x, y: x * y, output, attention_vector))
        # concat + conv
        V = reduce(lambda x, y: self.fusion2(torch.cat([x, y], dim=1)), V)

        return V

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=True, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class My(nn.Module):

    def __init__(self, num_classes):
        super(My, self).__init__()

        self.res1 = Residual(3, 64, stride=2)
        self.res2 = Residual(64, 64, stride=2)
        self.res3 = Residual(64, 128, stride=2)
        self.res4 = Residual(128, 256, stride=2)
        self.res5 = Residual(256, 256, stride=2)

        self.res1_sar = Residual(1, 64, stride=2)
        self.res1_sar2 = Residual(64, 64, stride=1)
        self.res2_sar = Residual(64, 64, stride=2)
        self.res2_sar2 = Residual(64, 64, stride=1)
        self.res3_sar = Residual(64, 128, stride=2)
        self.res3_sar2 = Residual(128, 128, stride=1)
        self.res4_sar = Residual(128, 256, stride=2)
        self.res4_sar2 = Residual(256, 256, stride=1)
        self.res5_sar = Residual(256, 256, stride=2)
        self.res5_sar2 = Residual(256, 256, stride=1)


        self.refu5 = SAGate(256, 256)
        self.refu4 = SAGate(256, 256)
        self.refu3 = SAGate(128, 128)
        self.refu2 = SAGate(64, 64)
        self.refu1 = SAGate(64, 64)

        self.defu5 = SAGate(256, 256)
        self.defu4 = SAGate(256, 256)
        self.defu3 = SAGate(128, 128)
        self.defu2 = SAGate(64, 64)
        self.defu1 = SAGate(64, 64)


        self.fu_sar_opt = SAGate(16, 16)

        self.restore5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU())
        self.restore4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.ReLU())
        self.restore4_sar = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.ReLU())
        self.restore4_sar = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.ReLU())
        self.restore3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU())
        self.restore3_sar = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU())
        self.restore2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
        )
        self.restore2_sar = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
        )
        self.restore1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=2, stride=2),
        )
        self.restore1_sar = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=2, stride=2),
        )

        self.final_fused = nn.Sequential(
            nn.Conv2d(16, num_classes, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

        initialize_weights(self)

    def forward(self, sar, opt):
        sar = sar.to(torch.float32)
        opt = opt.to(torch.float32)
        opt_en1 = self.res1(opt)
        opt_en2 = self.res2(opt_en1)
        opt_en3 = self.res3(opt_en2)
        opt_en4 = self.res4(opt_en3)
        opt_en5 = self.res5(opt_en4)

        sar_en1 = self.res1_sar(sar)
        sar_en1 = self.res1_sar2(sar_en1)
        sar_en1 = self.refu1(sar_en1, opt_en1)

        sar_en2 = self.res2_sar(sar_en1)
        sar_en2 = self.res2_sar2(sar_en2)
        sar_en2 = self.refu2(sar_en2, opt_en2)

        sar_en3 = self.res3_sar(sar_en2)
        sar_en3 = self.res3_sar2(sar_en3)
        sar_en3 = self.refu3(sar_en3, opt_en3)

        sar_en4 = self.res4_sar(sar_en3)
        sar_en4 = self.res4_sar2(sar_en4)
        sar_en4 = self.refu4(sar_en4, opt_en4)

        sar_en5 = self.res5_sar(sar_en4)
        sar_en5 = self.res5_sar2(sar_en5)
        sar_en5 = self.refu5(sar_en5, opt_en5)

        deopt_en5 = self.restore5(opt_en5)
        deopt_en4 = self.restore4(torch.cat([deopt_en5, opt_en4], 1))
        deopt_en3 = self.restore3(torch.cat([deopt_en4, opt_en3], 1))
        deopt_en2 = self.restore2(torch.cat([deopt_en3, opt_en2], 1))
        deopt_en1 = self.restore1(torch.cat([deopt_en2, opt_en1], 1))

        fu5 = self.defu5(sar_en5, opt_en5)
        desar_en5 = self.restore5(fu5)

        fu4 = self.defu4(sar_en4, deopt_en5)
        desar_en4 = self.restore4_sar(torch.cat([desar_en5, fu4], 1))

        fu3 = self.defu3(sar_en3, deopt_en4)
        desar_en3 = self.restore3_sar(torch.cat([desar_en4, fu3], 1))

        fu2 = self.defu2(sar_en2, deopt_en3)
        desar_en2 = self.restore2_sar(torch.cat([desar_en3, fu2], 1))

        fu1 = self.defu1(sar_en1, deopt_en2)
        desar_en1 = self.restore1_sar(torch.cat([desar_en2, fu1], 1))
        desar_opt = self.fu_sar_opt(desar_en1, deopt_en1)
        out = self.final_fused(desar_opt)

        return out

if __name__ == "__main__":
    model = My(num_classes=8).cuda()
    model.train()
    sar = torch.randn(2, 1, 256, 256).cuda()
    opt = torch.randn(2, 3, 256, 256).cuda()
    print(model)
    out = model(sar, opt)
    print("input:", sar.shape, opt.shape)
    print("output:", model(sar, opt).shape)
