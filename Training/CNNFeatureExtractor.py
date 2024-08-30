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


def load_FE_kwargs(number_of_actions: int, use_customCNN: bool = False) -> dict:
    """
    Loads the feature extractor kwargs for the model.

    If the user wants to use the CustomCNN, then the correct feature extractor class and kwargs are returned.
    If the user wants to use the Default NatureCNN, then only the features_dim is altered.

    :param number_of_actions: The number of actions the feature extractor will return. (features_dim)
    :param use_customCNN: Whether to use the CustomCNN or not.

    :return: A dictionary containing the policy's feature extractor kwargs.
    """
    if use_customCNN:
        return {'features_extractor_class': CNNFeatureExtractor,
                'features_extractor_kwargs': {'number_of_actions': number_of_actions}}

    return {'features_extractor_kwargs': {'features_dim': number_of_actions}}
