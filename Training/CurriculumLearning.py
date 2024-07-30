from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes import TrainAndLog_Callback
from Models import Doom_Models
import glob
import os


def CurriculumLearning(timesteps: int,
                       level: str,
                       model: Doom_Models,
                       callback: TrainAndLog_Callback,
                       render: bool,
                       log_name: str):

    latest_model = None
    last_model_location = f'Data/Train/train_{level}/{callback.get_formatted_datetime()}'

    for skill in range(1, 5):
        sub_level = f'{level}_s{skill}'
        print(f"\nTraining on sub-level: {sub_level}...")
        doom = ViZDoom_Gym(level=sub_level,
                           render=render,
                           reward_shaping=False,
                           curriculum=True)
        model.set_env(doom)
        model.learn(total_timesteps=timesteps,
                    callback=callback,
                    tb_log_name=log_name,
                    progress_bar=True,
                    reset_num_timesteps=False)

        model_location = glob.glob(f'{last_model_location}/*')

        if model_location:
            # Gets the latest model from the folder and continues training with it.
            latest_model = max(model_location, key=os.path.getctime)
            model.load(latest_model)
        else:
            print(f"No models found in {last_model_location} to load after training {sub_level}.")
            break

    return latest_model
