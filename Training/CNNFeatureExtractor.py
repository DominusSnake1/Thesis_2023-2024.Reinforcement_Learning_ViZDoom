from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from gymnasium.spaces import Box
import torch.nn as nn
import torch


class myCNN(nn.Module):
    def __init__(self, out_features: int = 512):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 30 * 40, 512)
        self.fc2 = nn.Linear(512, out_features)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 512):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = myCNN()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        features_extractor_class = kwargs.pop('features_extractor_class', CustomCNNFeatureExtractor)
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {'features_dim': 512})

        super(CustomCnnPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args, **kwargs,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs
        )
