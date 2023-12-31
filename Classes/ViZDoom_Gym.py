import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import vizdoom as vzd
import cv2


def grayscale(observation):
    gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    return gray.reshape((1, *gray.shape)).astype(np.uint8)


class ViZDoom_Gym(Env):
    def __init__(self, level, render=False):
        super().__init__()
        self.level = level
        self.game = vzd.DoomGame()
        self.game.set_doom_game_path('Utils/DOOM2.WAD')
        self.game.load_config(f'ViZDoom/scenarios/{level}.cfg')

        self.__level_adjustments()

        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init()
        self.observation_space = Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)
        self.action_space = Discrete(self.game.get_available_buttons_size())
        self.actions = np.identity(self.game.get_available_buttons_size(), dtype=np.uint8)

    def print_available_game_variables(self):
        print(self.game.get_available_game_variables())

    def __level_adjustments(self):
        if self.level == 'deadly_corridor':
            self.game.add_available_game_variable(vzd.DAMAGE_TAKEN)
            self.game.add_available_game_variable(vzd.HITCOUNT)
            self.game.add_available_game_variable(vzd.SELECTED_WEAPON_AMMO)

            setattr(self, 'damage_taken', 0)
            setattr(self, 'hitcount', 0)
            setattr(self, 'ammo', 52)

    def __reward_shaping(self, game_variables):
        if self.level == 'deadly_corridor':
            health, damage_taken, hitcount, ammo = game_variables

            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            return damage_taken_delta * 10 + hitcount_delta * 200 + ammo_delta * 5
        else:
            return 0

    def step(self, action):
        reward = self.game.make_action(self.actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = grayscale(state)

            info = self.game.get_state().game_variables
            reward += self.__reward_shaping(info)
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = 0

        done = self.game.is_episode_finished()
        info = {"info": info}
        truncated = False

        return state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return grayscale(state), 0

    def close(self):
        self.game.close()
