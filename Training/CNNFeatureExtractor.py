from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from Classes import ViZDoom_Gym
from gymnasium.spaces import Box
import torch.nn.functional as F
import torch.nn as nn
import torch


class myCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(myCNN, self).__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.output = nn.Linear(72384, features_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)

        x = x.view(batch_size, -1)
        x = self.output(x)
        return x


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = myCNN(observation_space, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        actions = 7
        super(CustomCnnPolicy, self).__init__(
            *args, **kwargs,
            features_extractor_class=CustomCNNFeatureExtractor,
            features_extractor_kwargs={'features_dim': actions},
        )
