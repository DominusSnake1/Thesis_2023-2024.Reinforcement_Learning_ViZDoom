from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes import TrainAndLog_Callback
from Models import Doom_Models


def CurriculumLearning(timesteps: int,
                       level: str,
                       model: Doom_Models,
                       callback: TrainAndLog_Callback,
                       render: bool,
                       log_name: str):

    first_level = f'{level}_s1'

    doom = ViZDoom_Gym(level=first_level,
                       render=render,
                       reward_shaping=False,
                       curriculum=True)
    model.set_env(doom)

    # For Doom Skill (difficulty) 2 to 5.
    for skill in range(2, 6):
        print(f"\nTraining on sub-level: {doom.level}...")

        model.learn(total_timesteps=timesteps,
                    callback=callback,
                    tb_log_name=log_name,
                    progress_bar=True,
                    reset_num_timesteps=False)

        next_level = f'{level}_s{skill}'
        doom = ViZDoom_Gym(level=next_level,
                           render=render,
                           reward_shaping=False,
                           curriculum=True)
        model.set_env(doom)

    return model
