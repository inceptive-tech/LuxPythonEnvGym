from turtle import undo

from luxai2021.game.cell import Cell
from luxai2021.game.position import Position
from luxai2021.game.game import Game
from cluster import Cluster


class ResourceClusterer:
    """
    Resource clusterer : gives a game a sets the clusters. Then computes, for each game state, the reward associated
    with
    """

    def __init__(self, game: Game) -> None:
        """
        :param game:
        """
        self.clusters = []
        #unclustered_resources = self.__build_unclustered_map(game.map.resources)
        unclustered_resources = game.map.resources.copy()

        while len(unclustered_resources) > 0:
            cur_cell = unclustered_resources.pop()
            cur_cluster = Cluster(cur_cell, game.map)
            self.clusters.append(cur_cluster)
            for cell in cur_cluster.cells:
                if cell in unclustered_resources:
                    unclustered_resources.remove(cell)


    def getReward(self, game: Game, team: int, unit_reward:float) -> float:
        res = 0
        for cur_cluster in self.clusters:
            ended_cluster = False
            for cur_cell in cur_cluster.cells:
                for adj in game.map.get_adjacent_cells(cur_cell):
                    if adj.city_tile is not None and adj.city_tile.team == team:
                        res = cur_cluster.reward_value(unit_reward)
                        ended_cluster = True
                    if ended_cluster:
                        break
                if ended_cluster:
                    break
        return res




