from Levels.Doom_Levels import Doom_Levels


def main():
    DOOM = Doom_Levels()

    if DOOM.level == 'basic':
        DOOM.basic()
    elif DOOM.level == 'defend_the_center':
        DOOM.defend_the_center()
    elif DOOM.level == 'deadly_corridor':
        DOOM.deadly_corridor()


if __name__ == '__main__':
    main()
