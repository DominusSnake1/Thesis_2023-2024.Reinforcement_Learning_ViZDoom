from Classes.ViZDoom_Gym import ViZDoom_Gym
import vizdoom as vzd


def basic(self: ViZDoom_Gym):
    self.game.add_available_game_variable(vzd.KILLCOUNT)

    setattr(self, 'killcount', 0)
    setattr(self, 'previous_ammo', 50)


def deadly_corridor(self: ViZDoom_Gym):
    self.game.add_available_game_variable(vzd.KILLCOUNT)
    self.game.add_available_game_variable(vzd.HITS_TAKEN)
    self.game.add_available_game_variable(vzd.ARMOR)
    self.game.add_available_game_variable(vzd.POSITION_X)

    setattr(self, 'killcount', 0)
    setattr(self, 'hits_taken', 0)
    setattr(self, 'X_position', 0)


def defend_the_center(self: ViZDoom_Gym):
    self.game.add_available_game_variable(vzd.KILLCOUNT)
    self.game.add_available_game_variable(vzd.HITS_TAKEN)

    setattr(self, 'prev_killcount', 0)
    setattr(self, 'prev_ammo', 26)
    setattr(self, 'prev_hits_taken', 0)


def defend_the_line(self: ViZDoom_Gym):
    self.game.add_available_game_variable(vzd.KILLCOUNT)
    self.game.add_available_game_variable(vzd.DAMAGE_TAKEN)
    self.game.add_available_game_variable(vzd.CAMERA_ANGLE)

    setattr(self, 'killcount', 0)
    setattr(self, 'damage_taken', 0)
    setattr(self, 'previous_camera_angle', 0)
