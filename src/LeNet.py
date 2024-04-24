from torch import nn


class LeNet(nn.Module):
    """
    LeNet-5
    Input: 1x32x32
    Output: 10
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.c2_1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.c2_2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.ReLU()
        )
        self.f4 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU()
        )
        self.f5 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        output = self.c1(img)
        x = self.c2_1(output)
        output = self.c2_2(output)
        output += x
        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output
