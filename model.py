from torch import nn
from torchinfo import summary

class TextClassificationModelV0(nn.Module):
    def __init__(self, input_shape:int, output_shape:int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_shape,8,3,stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8,16,3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*6*6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,output_shape)
        )
    def forward(self, x):
        return self.classifier(self.conv_block(x))  
    
if __name__ == "__main__":
    model = TextClassificationModelV0(input_shape=1,output_shape=80)
    print(summary(model, input_size=[4,1,48,48]))