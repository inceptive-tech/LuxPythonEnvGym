from torch.onnx.symbolic_opset9 import to

from luxai2021.game.cell import Cell
from luxai2021.game.game_map import GameMap
from luxai2021.game.resource import Resource
from luxai2021.game.game import Game


class Cluster:
    def __init__(self, cell: Cell, game_map: GameMap) -> None:
        """
        Inits a cluster associated to a cell, searching for all the cells containing resources associated with the given cell
        :param cell:
        """
        self.cells = [cell]
        #Search the cluster in a recursive way
        self.__search_cell(cell, game_map)
        self.resource_types = {Resource.Types.WOOD:0, Resource.Types.COAL:0, Resource.Types.URANIUM:0}
        for cell in self.cells:
            if cell.resource is not None:
                self.resource_types[cell.resource.type] += 1


    def reward_value(self, unit_reward: float, game: Game, team: int) -> float:
        # Reward 0.1 * unit_reward to 0.5 * unit_reward max
        # Gain 0.1 * unit_reward for each woods, 0.2 for each coal and 0.5 for uranium only if researched!
        res = 0.1 * unit_reward * self.resource_types[Resource.Types.WOOD]
        if game.state["teamStates"][team]["researched"]["coal"]:
            res += 0.2 * unit_reward * self.resource_types[Resource.Types.COAL]
        elif game.state["teamStates"][team]["researched"]["uranium"]:
            res += 0.5 * unit_reward * self.resource_types[Resource.Types.URANIUM]
        if res < 0.5 * unit_reward:
            return res
        else:
            return 0.5 * unit_reward


    def __search_cell(self,cell: Cell, game_map:GameMap):
        to_search = []
        for adj in game_map.get_adjacent_cells(cell):
            if adj.resource is not None and adj not in self.cells:
                self.cells.append(adj)
                self.__search_cell(adj, game_map)
