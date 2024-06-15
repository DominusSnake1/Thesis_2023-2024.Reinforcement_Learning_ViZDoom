from Classes.TrainAndLog_Callback import get_formatted_datetime
from Classes.ViZDoom_Gym import ViZDoom_Gym
from Models.Doom_Models import Doom_Models
import glob
import os


def CurriculumLearning(self: Doom_Models, level_name, model, callback):
    """
    Uses pre created levels (with increasing difficulty) to train a model.
    After a certain amount of training steps (50000), the level's difficulty increases accordingly.
    Then the same model is trained again.

    :param self: Object containing information about the model
    :param level_name: Name of the level
    :param model: Model to be trained
    :param callback: Callback function
    """
    sub_levels = [level_name + "_s1", level_name + "_s2", level_name + "_s3", level_name + "_s4"]
    base_model_location = f'Data/Train/train_{level_name}/{get_formatted_datetime()}'

    for sub_level in sub_levels:
        print(f"\nTraining on sub-level: {sub_level}...")
        env = ViZDoom_Gym(level=sub_level, reward_shaping=self.adjustments, render=self.render, curriculum=True)
        model.set_env(env)
        model.learn(total_timesteps=50000, callback=callback)

        model_location = glob.glob(f'{base_model_location}/*')

        if model_location:
            # Gets the latest model from the folder and continues training with it.
            latest_model = max(model_location, key=os.path.getctime)
            model.load(latest_model)
        else:
            print(f"No models found in {base_model_location} to load after training {sub_level}.")
            break
