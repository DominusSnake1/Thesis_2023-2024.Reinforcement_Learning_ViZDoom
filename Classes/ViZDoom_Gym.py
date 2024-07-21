from gymnasium.spaces import Discrete, Box
from gymnasium import Env
import vizdoom as vzd
import numpy as np
from icecream import ic


class ViZDoom_Gym(Env):
    def __init__(self, level, render=False, reward_shaping=False, curriculum=False):
        super().__init__()
        self.level = level
        self.reward_shaping = reward_shaping
        self.use_curriculum = curriculum
        self.game = vzd.DoomGame()
        self.game.set_doom_game_path('Other/DOOM2.WAD')

        if self.use_curriculum:
            self.level = self.level[:-3]
            cfg = f'Levels/LevelsCurriculum/{level}.cfg'
        else:
            cfg = f'ViZDoom/scenarios/{level}.cfg'

        self.game.load_config(cfg)

        if self.reward_shaping:
            from Training import LevelAdjustments
            getattr(LevelAdjustments, self.level)(self)

        self.game.set_window_visible(render)

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.observation_space = Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)
        self.action_space = Discrete(self.game.get_available_buttons_size())
        self.actions = np.identity(self.game.get_available_buttons_size(), dtype=np.uint8)
        self.game.init()

    def print_available_game_variables(self):
        print(self.game.get_available_game_variables())

    def __reward_shaping(self, game_variables: np.ndarray) -> int:
        from Training import RewardShaping

        return getattr(RewardShaping, self.level)(self, game_variables)

    def step(self, action):
        reward = self.game.make_action(self.actions[action], 4)

        state = self.game.get_state()

        if state is not None:
            screen_buffer = state.screen_buffer
            # state = grayscale(state)

            game_variables = state.game_variables

            if self.reward_shaping:
                reward += self.__reward_shaping(game_variables)
        else:
            screen_buffer = np.zeros(self.observation_space.shape, dtype=np.uint8)
            game_variables = 0

        done = self.game.is_episode_finished()
        info = {"info": game_variables}
        truncated = False

        return screen_buffer, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        screen_buffer = self.game.get_state().screen_buffer
        return screen_buffer, 0
        # return grayscale(state), 0

    def close(self):
        self.game.close()
