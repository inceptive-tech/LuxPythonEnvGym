import unittest

from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game
from luxai2021.game.game_map import GameMap


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # Given
        config = dict(LuxMatchConfigs_Default)
        config["seed"] = 10
        game = Game(config)
        game.spawn_city_tile(0, 5, 5)
        game_map = GameMap(config)
        game_map.generate_map(game)
        print()
        print(game_map.get_map_string())
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
