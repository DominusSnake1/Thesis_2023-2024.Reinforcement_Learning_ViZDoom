from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
import torchvision.models as models
import torch.nn as nn
import torch


class ResNet50FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 2048):
        super(ResNet50FeatureExtractor, self).__init__(observation_space, features_dim)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(2048, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.resnet(observations)
        return self.fc(features)


class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        features_extractor_class = kwargs.pop('features_extractor_class', ResNet50FeatureExtractor)
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {'features_dim': 512})

        super(CustomCnnPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args, **kwargs,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs
        )
