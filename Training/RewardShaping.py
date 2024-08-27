from Classes.ViZDoom_Gym import ViZDoom_Gym


def deadly_corridor(self: ViZDoom_Gym, game_variables: list) -> float:
    """
    Custom reward function partially generated by ChatGPT-4o.

    :return: Extra reward according to the coefficients.
    """
    # Unpack the game variables.
    health, killcount, hits_taken, armor, pos_x = game_variables

    # Initializations.
    REWARD = 0

    MOVED_CLOSER_TO_VEST = pos_x > self.X_position
    STAYED_IN_PLACE = pos_x == self.X_position
    TOOK_DAMAGE = hits_taken > self.hits_taken
    GOT_KILL = killcount > self.killcount
    GOT_ARMOR = armor > 0
    IS_DEAD = health <= 0

    # Get (+) reward when the agent moves closer to the vest.
    if MOVED_CLOSER_TO_VEST:
        REWARD += 50

    # Get (+) reward when the agent gets a kill.
    if GOT_KILL:
        REWARD += 50

    # Get (+) reward if the agent took no extra damage this step.
    if not TOOK_DAMAGE:
        REWARD += 5

    # Get (+) reward if the agent got the armor.
    if GOT_ARMOR:
        REWARD += 1000

    # Get (-) if the agent dies.
    if IS_DEAD:
        REWARD -= 50

    # Get (-) reward only when the agent gets hit.
    if TOOK_DAMAGE:
        REWARD -= 15

    # Get (-) reward only when the agent moves away from the vest.
    if not MOVED_CLOSER_TO_VEST:
        REWARD -= 50

    # Get (-) reward when the agent stays in place.
    if STAYED_IN_PLACE:
        REWARD -= 5

    # Get (-) reward per timestep.
    REWARD += 1

    # Update the values for the next step.
    self.killcount = killcount
    self.hits_taken = hits_taken
    self.X_position = pos_x

    return REWARD


def defend_the_center(self: ViZDoom_Gym, game_variables: list) -> float:
    # Unpack game variables.
    ammo, health, killcount = game_variables

    # Initial reward.
    REWARD = 0

    SPENT_AMMO = ammo < self.previous_ammo
    DAMAGED = health < self.previous_health
    GOT_KILL = killcount > self.killcount

    # Reward for killing a monster.
    if GOT_KILL:
        REWARD += 10

    # Penalty for wasting ammunition without getting any kills.
    if SPENT_AMMO and not GOT_KILL:
        REWARD -= 1

    if DAMAGED:
        REWARD -= -5

    # Update variables for the next step.
    self.previous_health = health
    self.killcount = killcount
    self.previous_ammo = ammo

    return REWARD


def deathmatch(self: ViZDoom_Gym, game_variables: list) -> float:
    pass
