from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes import TrainAndLog_Callback
from gymnasium import Env, spaces
from Models import Doom_Models
import numpy as np


def CurriculumLearning(timesteps: int,
                       level: str,
                       default_skill: int,
                       model: Doom_Models,
                       callback: TrainAndLog_Callback,
                       render: bool,
                       display_rewards: bool,
                       reward_shaping: bool,
                       log_name: str):

    doom = ViZDoom_Gym(level=level,
                       difficulty=1,
                       render=render,
                       display_rewards=display_rewards,
                       reward_shaping=reward_shaping,
                       curriculum=True)

    model.set_env(doom)

    print(f"\nTraining on '{doom.level}' with 1/{default_skill} doom_skill...")
    model.learn(total_timesteps=timesteps,
                callback=callback,
                tb_log_name=log_name,
                progress_bar=True,
                reset_num_timesteps=False)

    doom.close()

    # For Doom Skill (difficulty) 2 to the default difficulty for that level.
    for skill in range(2, (default_skill + 1)):
        doom = ViZDoom_Gym(level=level,
                           difficulty=skill,
                           render=render,
                           display_rewards=display_rewards,
                           reward_shaping=reward_shaping,
                           curriculum=True)

        model.set_env(doom)

        print(f"\nTraining on '{doom.level}' with {skill}/{default_skill} doom_skill...")

        progression_timesteps = int(timesteps / 10)

        model.learn(total_timesteps=progression_timesteps,
                    callback=callback,
                    tb_log_name=log_name,
                    progress_bar=True,
                    reset_num_timesteps=False)

        doom.close()


class Blank_Env(Env):
    def __init__(self, actions):
        super(Blank_Env, self).__init__()
        self.action_space = spaces.Discrete(actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)

        def reset(self):
            return np.zeros(self.observation_space.shape)

        def step(self, action):
            return np.zeros(self.observation_space.shape), 0, True, False, {}

        def render(self):
            pass
