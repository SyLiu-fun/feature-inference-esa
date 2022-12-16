from torch import nn


# create a LR model
class LogisticRegressionModel(nn.Module):
    def __init__(self, feature_num, class_num):
        super(LogisticRegressionModel, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(feature_num, class_num, False),
        )

    def forward(self, x):
        return self.dense(x)
