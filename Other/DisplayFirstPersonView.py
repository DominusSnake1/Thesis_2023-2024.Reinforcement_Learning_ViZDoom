from Other.CMD import level_selector
from argparse import ArgumentParser
import vizdoom as vzd

parser = ArgumentParser()

parser.add_argument(
    dest="scenario",
    help="The scenario whose first person view will be displayed",
)

args = parser.parse_args()
level = level_selector(args.scenario)

game = vzd.DoomGame()

game.load_config(level)
game.set_render_hud(True)

game.set_screen_resolution(vzd.ScreenResolution.RES_1024X576)


# Disables game window (FPP view), we just want to see the automap.
game.set_window_visible(True)

game.init()

# Not needed for the first episode but the loop is nicer.
game.new_episode()

# Gets the state
state = game.get_state()

input("Press Enter to exit...")

game.close()
