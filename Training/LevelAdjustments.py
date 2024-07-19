from Classes.ViZDoom_Gym import ViZDoom_Gym
import vizdoom as vzd


def deadly_corridor(self: ViZDoom_Gym):
    self.game.add_available_game_variable(vzd.KILLCOUNT)
    self.game.add_available_game_variable(vzd.SELECTED_WEAPON_AMMO)
    self.game.set_living_reward(-1)

    setattr(self, 'killcount', 0)
    setattr(self, 'ammo', 52)
