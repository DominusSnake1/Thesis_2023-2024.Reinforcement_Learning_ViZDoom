from Levels.Doom_Levels import Doom_Levels


def main():
    DOOM = Doom_Levels()
    selected_level = getattr(DOOM, DOOM.level)

    selected_level()


if __name__ == '__main__':
    main()
