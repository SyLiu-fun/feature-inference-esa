from torch import nn


# create a LR model
class LogisticRegressionModel(nn.model):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(10, 2, False),
            nn.Softmax()
        )

    def forward(self, x):
        return self.dense(x)
