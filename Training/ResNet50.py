from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
import torchvision.models as models
from stable_baselines3 import PPO
import torch.nn as nn
import torch


class ResNet50FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 2048):
        super(ResNet50FeatureExtractor, self).__init__(observation_space, features_dim)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

        # Freeze ResNet layers if you don't want them to be trainable
        for param in self.resnet.parameters():
            param.requires_grad = False

        # If you want to add a trainable layer after ResNet
        self.fc = nn.Linear(2048, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # Ensure ResNet is not trained if frozen
            features = self.resnet(observations)
        return self.fc(features)


class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs,
                                              features_extractor_class=ResNet50FeatureExtractor,
                                              features_extractor_kwargs={'features_dim': 512})
