from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import torchvision.transforms as transforms
import torchvision.models as models
from gymnasium.spaces import Box
import torch.nn as nn
import torch

preprocessImage = transforms.Compose([
    # Converts a tensor or numpy array to a PIL Image.
    transforms.ToPILImage(),
    # Resizes the image to 256x256 pixels.
    transforms.Resize(256),
    # Crops the center 224x224 pixels from the image.
    transforms.CenterCrop(224),
    # Converts the PIL Image to a PyTorch tensor.
    transforms.ToTensor(),
    # Normalizes the tensor using the specified mean and standard deviation.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ResNet50FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 2048):
        super(ResNet50FeatureExtractor, self).__init__(observation_space, features_dim)

        # Loads the pre-trained ResNet50 model from torchvision.
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Removes the final fully connected layer from ResNet50.
        modules = list(resnet50.children())[:-1]
        # Creates a sequential container of the ResNet50 layers.
        self.resnet50 = nn.Sequential(*modules)

        # Freezes the weights of ResNet50 so that they are not updated during training.
        for p in self.resnet50.parameters():
            p.requires_grad = False

        # Defines a fully connected layer.
        self.fc = nn.Linear(2048, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Repeats grayscale images across the color channels to make them
        # compatible with the ResNet50 model which expects 3-channel RGB images.
        observations = observations.repeat(1, 3, 1, 1)[:, :3, :, :]

        # Applies preprocessing to each image in the batch and stacks them into a single tensor.
        observations = torch.stack([preprocessImage(obs) for obs in observations])

        # Moves the tensor to the same device (CPU or GPU) as the model parameters.
        observations = observations.to(next(self.parameters()).device)

        with torch.no_grad():
            # Passes the images through ResNet50 to extract features.
            features = self.resnet50(observations)
            # Flattens the output features from ResNet50.
            features = features.flatten(start_dim=1)

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
