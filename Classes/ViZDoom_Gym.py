from gymnasium.spaces import Discrete, Box
from gymnasium import Env
import vizdoom as vzd
import numpy as np
import os.path
from icecream import ic


def process_observation(observation):
    obs = observation.astype(np.float32)
    obs /= 255.0
    obs -= 0.5
    return obs


class ViZDoom_Gym(Env):
    def __init__(self,
                 level: str,
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

        cfg = f'ViZDoom/scenarios/{level}.cfg'

        if self.use_curriculum and os.path.exists(f'Levels/LevelsCurriculum/{level}.cfg'):
            cfg = f'Levels/LevelsCurriculum/{level}.cfg'

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

    def __reward_shaping(self, game_variables: np.ndarray) -> float:
        from Training import RewardShaping

        return getattr(RewardShaping, self.level)(self, game_variables)

    def step(self, action):
        # Get the BASE Reward from the level depending on the action taken.
        base_reward = round(self.game.make_action(self.actions[action], tics=5), 3)

        # Extract the current state of the game.
        state = self.game.get_state()
        # Get whether the episode is finished.
        done = self.game.is_episode_finished()

        # If there is no current_state, enter...
        if state is None:
            # Initialize the observation space to zeroes.
            self.observation_space = np.zeros(self.observation_space.shape, dtype=np.uint8)
            # Returns Observation, Reward, Done, Truncated, Info
            return self.observation_space, base_reward, done, False, {"info": 0}

        # Normalize from [0.0 , 1.0] to [-0.5 , 0.5].
        self.observation_space = process_observation(state.screen_buffer)
        # Extract the game variables from the current state.
        game_variables = state.game_variables

        # Initialize the TOTAL Reward (BASE + EXTRA), in case of Reward Shaping.
        total_reward = base_reward
        # Initialize the EXTRA Reward, in case of Reward Shaping.
        extra_reward = 0

        # If selected Technique features Reward Shaping, enter...
        if self.reward_shaping:
            # Get the EXTRA Reward from the custom Reward Function.
            extra_reward = self.__reward_shaping(game_variables)
            # Add the EXTRA Reward to the BASE Reward.
            total_reward += extra_reward

        # If the user wants the rewards to be displayed, enter...
        if self.display_rewards:
            # Display the BASE Reward and the EXTRA Reward (in case of Reward Shaping)
            self.__display_rewards(base_reward, extra_reward)

        # Returns Observation, Reward, Done, Truncated, Info
        return self.observation_space, total_reward, done, False, {"info": game_variables}

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

    def __display_rewards(self, base_reward, extra_reward):
        """
        Displays the BASE reward for each step.

        If the used technique features Reward Shaping, then the EXTRA reward is also displayed.

        :param base_reward: The reward from the level.
        :param extra_reward: The reward from the custom reward function.
        """
        print(f"BASE Reward: {base_reward}")

        if self.reward_shaping:
            print(f"\tEXTRA Reward: {extra_reward}")

