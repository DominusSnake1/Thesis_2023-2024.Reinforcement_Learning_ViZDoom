import numpy as np
from vizdoom import *
import random
import time

if __name__ == '__main__':
    game = DoomGame()
    game.load_config('ViZDoom/scenarios/basic.cfg')
    game.init()

    actions = np.identity(3, dtype=np.uint8)

    episodes = 10
    for episode in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            info = state.game_variables
            reward = game.make_action(random.choice(actions))
            print("Reward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)