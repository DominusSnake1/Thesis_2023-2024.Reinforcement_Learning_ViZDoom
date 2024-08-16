from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes import TrainAndLog_Callback
from Models import Doom_Models


def CurriculumLearning(timesteps: int,
                       level: str,
                       model: Doom_Models,
                       callback: TrainAndLog_Callback,
                       render: bool,
                       display_rewards: bool,
                       reward_shaping: bool,
                       log_name: str):

    first_level = f'{level}_s1'

    doom = ViZDoom_Gym(level=first_level,
                       render=render,
                       display_rewards=display_rewards,
                       reward_shaping=reward_shaping,
                       curriculum=True)

    model.set_env(doom)

    print(f"\nTraining on sub-level: {doom.level}...")
    model.learn(total_timesteps=timesteps,
                callback=callback,
                tb_log_name=log_name,
                progress_bar=True,
                reset_num_timesteps=False)

    doom.close()

    # For Doom Skill (difficulty) 2 to 5.
    for skill in range(2, 6):
        next_level = f'{level}_s{skill}'

        doom = ViZDoom_Gym(level=next_level,
                           render=render,
                           display_rewards=display_rewards,
                           reward_shaping=reward_shaping,
                           curriculum=True)

        model.set_env(doom)

        print(f"\nTraining on sub-level: {doom.level}...")

        progression_timesteps = int(timesteps/10)

        model.learn(total_timesteps=progression_timesteps,
                    callback=callback,
                    tb_log_name=log_name,
                    progress_bar=True,
                    reset_num_timesteps=False)

        doom.close()
