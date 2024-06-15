from Classes.ViZDoom_Gym import ViZDoom_Gym


def deadly_corridor(self: ViZDoom_Gym, game_variables: list):
    # The agent gets reward (0 - 1312 pts) according to his distance from the vest.
    # The agent gets reward (1000 pts) if he reaches the vest.

    health, damage_taken, killcount, ammo = game_variables
    print(f"HEALTH:{health}, DMG_TAKEN:{damage_taken}, KILLS:{killcount}, AMMO:{ammo}")

    damage_taken_delta = damage_taken - self.damage_taken
    self.damage_taken = damage_taken
    killcount_delta = killcount - self.killcount
    self.killcount = killcount

    if ammo == 8:
        self.ammo = ammo

    ammo_delta = self.ammo - ammo
    self.ammo = ammo

    damage_taken_coef = -10
    killcount_coef = 200
    ammo_coef = -2

    return (damage_taken_delta * damage_taken_coef) + (killcount_delta * killcount_coef) + (ammo_delta * ammo_coef)
