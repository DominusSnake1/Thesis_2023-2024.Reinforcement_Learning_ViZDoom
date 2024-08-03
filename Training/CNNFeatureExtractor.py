from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
import torch.nn.functional as F
from icecream import ic
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

        # Define the first convolutional layer, followed by a batch normalization layer and a pooling layer.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the second convolutional layer, followed by a batch normalization layer and a pooling layer.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the third convolutional layer, followed by a batch normalization layer and a pooling layer.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the fourth convolutional layer, followed by a batch normalization layer and a pooling layer.
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define a flattening layer to convert 2D feature maps to a 1D feature vector.
        self.flatten = nn.Flatten()

        # Define a dropout layer to prevent overfitting.
        self.dropout = nn.Dropout(p=0.5)

        # Define a fully connected layer to output the number of actions.
        self.output = nn.Linear(34048, number_of_actions)  # Adjust the input size of this layer

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        :param observations: A batch of observations from the environment.

        :return: The output logits for the number of actions.
        """
        # Apply the first convolutional layer, followed by batch normalization, ReLU activation, and pooling.
        x = F.leaky_relu(self.bn1(self.conv1(observations)))
        x = self.pool1(x)

        # Apply the second convolutional layer, followed by batch normalization, ReLU activation, and pooling.
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Apply the third convolutional layer, followed by batch normalization, ReLU activation, and pooling.
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Apply the fourth convolutional layer, followed by batch normalization, ReLU activation, and pooling.
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Flatten the feature maps into a 1D vector.
        x = self.flatten(x)

        # Apply the dropout layer.
        x = self.dropout(x)

        # Apply the fully connected layer to get the final output logits.
        x = self.output(x)
        return x
