from Classes.ViZDoom_Gym import ViZDoom_Gym


def deadly_corridor(self: ViZDoom_Gym, game_variables: list) -> float:
    """
    The agent gets reward (0 - 1312 pts) according to his distance from the vest.
    The agent gets reward (1000 pts) if he reaches the vest.
    The agent loses 100 pts if he dies.

    :return: Extra reward according to the coefficients.
    """
    # Unpack the game variables.
    health, killcount, ammo = game_variables
    print(f"HEALTH:{health}, KILLS:{killcount}, AMMO:{ammo}")

    # Initialise the deltas.
    killcount_delta = 0
    ammo_delta = 0

    # Calculate the delta only when the agent gets a kill.
    if killcount > self.killcount:
        killcount_delta = killcount - self.killcount

    if ammo == 8:
        self.ammo = 8

    # Calculate the delta only when the agent spends a bullet.
    if self.ammo > ammo:
        ammo_delta = self.ammo - ammo

    self.killcount = killcount
    self.ammo = ammo

    # Reward agent for getting kills.
    killcount_coef = 150
    # Punish agent for spending bullets.
    ammo_coef = -1

    return (killcount_delta * killcount_coef) + (ammo_delta * ammo_coef)
