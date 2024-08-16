from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium.spaces import Box
import torch.nn.functional as F
import gymnasium as gym
import torch.nn as nn
import torch


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, number_of_actions: int):
        """
        Initialize the CNN feature extractor.

        :param observation_space: The observation space from the environment.
        :param number_of_actions: The number of possible actions the agent can take.
        """
        super(CNNFeatureExtractor, self).__init__(observation_space, number_of_actions)

        height = observation_space.shape[1]
        width = observation_space.shape[2]

        # Define the first convolutional layer, followed by a batch normalization layer and a pooling layer.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=4)

        height //= 4
        width //= 4

        # Define the second convolutional layer, followed by a batch normalization layer and a pooling layer.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=4)

        height //= 4
        width //= 4

        # Define a fully connected layer to output the number of actions.
        self.output = nn.Linear(height * width * 16, number_of_actions)  # Adjust the input size of this layer

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        :param observations: A batch of observations from the environment.

        :return: The output logits for the number of actions.
        """
        batch_size = observations.shape[0]

        # Apply the first convolutional layer, followed by batch normalization, ReLU activation, and pooling.
        x = F.leaky_relu(self.bn1(self.conv1(observations)))
        x = self.pool1(x)

        # Apply the second convolutional layer, followed by batch normalization, ReLU activation, and pooling.
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.view(batch_size, -1)

        # Apply the fully connected layer to get the final output logits.
        x = self.output(x)
        return x


class CustomCNN_Policy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 lr_schedule,
                 **kwargs):
        super(CustomCNN_Policy, self).__init__(observation_space,
                                               action_space,
                                               lr_schedule,
                                               features_extractor_class=CNNFeatureExtractor,
                                               **kwargs)
