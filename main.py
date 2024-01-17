from Levels.Doom_Levels import Doom_Levels


def main():
    DOOM = Doom_Levels()

    if DOOM.level == 'basic':
        DOOM.basic()
    elif DOOM.level == 'defend_the_center':
        DOOM.defend_the_center()
    elif DOOM.level == 'deadly_corridor':
        DOOM.deadly_corridor()
    elif DOOM.level == 'deathmatch':
        DOOM.deathmatch()
    elif DOOM.level == 'defend_the_line':
        DOOM.defend_the_line()
    elif DOOM.level == 'health_gathering':
        DOOM.health_gathering()
    elif DOOM.level == 'health_gathering_supreme':
        DOOM.health_gathering_supreme()
    elif DOOM.level == 'my_way_home':
        DOOM.my_way_home()
    elif DOOM.level == 'predict_position':
        DOOM.predict_position()
    elif DOOM.level == 'take_cover':
        DOOM.take_cover()


if __name__ == '__main__':
    main()
