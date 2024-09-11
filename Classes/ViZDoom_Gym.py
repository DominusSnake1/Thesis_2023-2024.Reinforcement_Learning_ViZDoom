from gymnasium.spaces import Discrete, Box
from gymnasium import Env
import vizdoom as vzd
import numpy as np
import os.path


def process_observation(observation):
    obs = observation.astype(np.float32)
    obs /= 255.0
    obs -= 0.5
    return obs


class ViZDoom_Gym(Env):
    def __init__(self,
                 level: str,
                 difficulty: int,
                 render: bool = False,
                 display_rewards: bool = False,
                 reward_shaping: bool = False,
                 curriculum: bool = False):
        super().__init__()

        self.level = level
        self.display_rewards = display_rewards
        self.reward_shaping = reward_shaping
        self.use_curriculum = curriculum
        self.game = vzd.DoomGame()
        self.game.set_doom_game_path('Other/DOOM2.WAD')

        self.game.load_config(f'ViZDoom/scenarios/{self.level}.cfg')

        self.game.set_doom_skill(difficulty)

        if self.reward_shaping:
            self.prepare_reward_shaping()

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.observation_space = Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)
        self.action_space = Discrete(self.game.get_available_buttons_size())
        self.actions = np.identity(self.game.get_available_buttons_size(), dtype=np.uint8)

        self.game.set_window_visible(render)

        self.game.init()

    def prepare_reward_shaping(self):
        from Training import LevelAdjustments
        level = self.level

        if self.use_curriculum:
            level = level[:-3]

        return getattr(LevelAdjustments, level)(self)

    def __reward_shaping(self, game_variables: np.ndarray) -> float:
        from Training import RewardShaping
        level = self.level

        if self.use_curriculum:
            level = level[:-3]

        return getattr(RewardShaping, level)(self, game_variables)

    def step(self, chosen_action):
        # Extract the current state of the game.
        state = self.game.get_state()

        # Get whether the episode is finished.
        done = self.game.is_episode_finished()

        # If there is no current_state, enter...
        if state is None:
            # Initialize the observation space to zeroes.
            self.observation_space = np.zeros(self.observation_space.shape, dtype=np.uint8)
            # Returns Observation, Reward, Done, Truncated, Info
            return self.observation_space, 0, done, False, {"info": 0}

        # Normalize from int [0 , 255] to float [-0.5 , 0.5].
        self.observation_space = process_observation(state.screen_buffer)

        self.game.set_action(action=self.actions[chosen_action])
        self.game.advance_action(tics=4)

        # Extract the game variables from the current state.
        game_variables = state.game_variables

        # Calculates the reward with the Default or the Custom Reward Function.
        reward = self.calculate_reward(game_variables=game_variables)

        # If the user wants the rewards to be displayed, enter...
        if self.display_rewards:
            # Display the BASE Reward and the CUSTOM Reward (in case of Reward Shaping)
            self.__display_rewards(reward)

        # Returns Observation, Reward, Done, Truncated, Info
        return self.observation_space, reward, done, False, {"info": game_variables}

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state()
        observation = process_observation(state.screen_buffer)
        return observation, 0

    def close(self):
        """
        Closes the game.
        """
        self.game.close()

    def calculate_reward(self, game_variables):
        """
        Calculates the reward for each step depending on whether the training uses Reward Shaping or not.
        If Reward Shaping is used, then a Custom Reward function is utilized, overwriting the default reward function.

        :param game_variables: The game variables of the environment
        """
        # If reward shaping is used, enter...
        if self.reward_shaping:
            # Get the reward from the Custom Reward function.
            return self.__reward_shaping(game_variables)

        # Get the BASE Reward from the level depending on the action taken.
        return round(self.game.get_last_reward(), 3)

    def __display_rewards(self, reward):
        """
        Displays the BASE reward for each step.

        If the used technique features Reward Shaping, then the EXTRA reward is also displayed.

        :param reward: The reward from the level.
        """
        if not self.reward_shaping:
            print(f"BASE Reward: {reward}")
            return

        print(f"CUSTOM Reward: {reward}")
        return
