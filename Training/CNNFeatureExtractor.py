from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
import torch.nn.functional as F
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

        # Define the first convolutional and pooling layer.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the second convolutional and pooling layer.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the third convolutional and pooling layer.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define a flattening layer to convert 2D feature maps to 1D feature vector.
        self.flatten = nn.Flatten()

        # Define a fully connected layer to output the number of actions
        self.output = nn.Linear(72384, number_of_actions)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        :param observations: A batch of observations from the environment.

        :return: The output logits for the number of actions.
        """
        # Apply the first convolutional layer, followed by ReLU activation and pooling.
        x = F.relu(self.conv1(observations))
        x = self.pool1(x)

        # Apply the second convolutional layer, followed by ReLU activation and pooling.
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Apply the third convolutional layer, followed by ReLU activation and pooling.
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the feature maps into a 1D vector.
        x = self.flatten(x)

        # Apply the fully connected layer to get the final output logits.
        x = self.output(x)
        return x
