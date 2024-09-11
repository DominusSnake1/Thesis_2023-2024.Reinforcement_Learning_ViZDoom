from argparse import ArgumentParser
from CMD import level_selector
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
game.set_doom_game_path('Other/DOOM2.WAD')
game.set_screen_resolution(vzd.ScreenResolution.RES_1280X1024)
game.set_window_visible(True)
game.init()

input("Press 'Anything' to exit...")

game.close()
