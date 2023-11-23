import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import vizdoom as vzd
import cv2


def grayscale(observation):
    gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    return gray


class ViZDoom_Gym(Env):
    def __init__(self, level, render=False):
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(f'ViZDoom/scenarios/{level}.cfg')

        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init()
        self.observation_space = Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = grayscale(state)
            info = self.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        done = self.game.is_episode_finished()

        return state, reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return grayscale(state)

    def close(self):
        self.game.close()
