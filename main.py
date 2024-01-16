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


if __name__ == '__main__':
    main()
