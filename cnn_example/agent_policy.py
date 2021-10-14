import sys
from functools import partial  # pip install functools

import numpy as np
from gym import spaces

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS

from rewards.resource_clusterer import ResourceClusterer


# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)


def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if (u.get_cargo_space_left() >= resource_amount and
                                        target_unit.get_cargo_space_left() >= resource_amount):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u

                                elif (target_unit.get_cargo_space_left() >= resource_amount):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass

                                elif (u.get_cargo_space_left() > target_unit.get_cargo_space_left()):
                                    # Change targets, because neither target can accept all our resources and 
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u

    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()

    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)


########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART),
            # Transfer to nearby cart
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER),
            # Transfer to nearby worker
            SpawnCityAction,
            # PillageAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            # SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))

        # Observation space: (Basic minimum for a miner agent)
        # The 20 features used are
        #
        # 0) Whether is this the unit making the decision
        #
        # 1) Unit cargo level
        #
        # 2) Existence of self unit(excluding the decision - making unit)
        #
        # 3) Self unit cooldown level
        #
        # 4) Self unit cargo level
        #
        # 5) Existence of opponent unit
        #
        # 6) Opponent unit cooldown level
        #
        # 7) Opponent unit cargo level
        #
        # 8) Existence of self city
        #
        # 9) Self city tile night survival duration
        #
        # 10) Existence of opponent city tile
        #
        # 11) Opponent city tile night survival duration
        #
        # 12) Resource wood level
        #
        # 13) Resource coal level
        #
        # 14) Resource  uranium level
        #
        # 15) Self research point
        #
        # 16) Opponent research point
        #
        # 17) Day night cycle number
        #
        # 18) Current turn number
        # 19) Whether is it out of bounds in the map
        #
        # 20) Whether is this the city_tile making the decision
        self.observation_shape = (21, 32, 32)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """

        # Observation space: (Basic minimum for a miner agent)
        # The 20 features used are
        #
        # 0) Whether is this the unit making the decision
        #
        # 1) Unit cargo level
        #
        # 2) Existence of self unit(excluding the decision - making unit)
        #
        # 3) Self unit cooldown level
        #
        # 4) Self unit cargo level
        #
        # 5) Existence of opponent unit
        #
        # 6) Opponent unit cooldown level
        #
        # 7) Opponent unit cargo level
        #
        # 8) Existence of self city
        #
        # 9) Self city tile night survival duration
        #
        # 10) Existence of opponent city tile
        #
        # 11) Opponent city tile night survival duration
        #
        # 12) Resource wood level
        #
        # 13) Resource coal level
        #
        # 14) Resource  uranium level
        #
        # 15) Self research point
        #
        # 16) Opponent research point
        #
        # 17) Day night cycle ratio
        #
        # 18) Current turn number
        # 19) Whether is it out of bounds in the map
        #
        # 20) Whether is this the city_tile making the decision
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        x_shift = (32 - game.map.width) // 2
        y_shift = (32 - game.map.height) // 2

        # Update the type of this object
        #   0) Whether is this the unit making the decision
        if unit is not None:
            obs[0][x_shift + unit.pos.x][y_shift + unit.pos.y] = 255
        # 20) Whether is this the city_tile making the decision
        if city_tile is not None:
            obs[20][x_shift + city_tile.pos.x][y_shift + city_tile.pos.y] = 255

        # Units channels
        for t in [team, (team + 1) % 2]:
            for u in game.state["teamStates"][t]["units"].values():
                # 1) Unit cargo level
                obs[1][x_shift + u.pos.x][y_shift + u.pos.y] = ((100 - u.get_cargo_space_left()) / 100) * 255
                if t == team:
                    # 2) Existence of self unit(excluding the decision - making unit)
                    obs[2][x_shift + u.pos.x][y_shift + u.pos.y] = 255
                    # 3) Self unit cooldown level
                    obs[3][x_shift + u.pos.x][y_shift + u.pos.y] = (u.cooldown / 6) * 255
                    # 4) Self unit cargo level
                    obs[4][x_shift + u.pos.x][y_shift + u.pos.y] = ((100 - u.get_cargo_space_left()) / 100) * 255
                else:
                    # 5) Existence of opponent unit
                    obs[5][x_shift + u.pos.x][y_shift + u.pos.y] = 255
                    # 6) Opponent unit cooldown level
                    obs[6][x_shift + u.pos.x][y_shift + u.pos.y] = (u.cooldown / 6) * 255
                    # 7) Opponent unit cargo level
                    obs[7][x_shift + u.pos.x][y_shift + u.pos.y] = ((100 - u.get_cargo_space_left()) / 100) * 255

        # City channels
        # 8) Existence of self city
        for city in game.cities.values():
            fuel = city.fuel
            light_upkeep = city.get_light_upkeep()
            city_survival = (min(fuel / light_upkeep, 10) / 10) * 255
            for cell in city.city_cells:
                if city.team == team:
                    # 8) Existence of self city
                    obs[8][x_shift + cell.pos.x][y_shift + cell.pos.y] = 255
                    # 9) Self city tile night survival duration
                    obs[9][x_shift + cell.pos.x][y_shift + cell.pos.y] = city_survival
                else:
                    # 10) Existence of opponent city tile
                    obs[10][x_shift + cell.pos.x][y_shift + cell.pos.y] = 255
                    # 11) Opponent city tile night survival duration
                    obs[11][x_shift + cell.pos.x][y_shift + cell.pos.y] = city_survival

        # Resource levels
        for cell in game.map.resources:
            # 12) Resource wood level
            if cell.resource.type == Constants.RESOURCE_TYPES.WOOD:
                obs[12][x_shift + cell.pos.x][y_shift + cell.pos.y] = cell.resource.amount * 255 / 800
            # 13) Resource coal level
            elif cell.resource.type == Constants.RESOURCE_TYPES.COAL:
                obs[13][x_shift + cell.pos.x][y_shift + cell.pos.y] = cell.resource.amount * 255 / 800
            # 14) Resource  uranium level
            else:
                obs[14][x_shift + cell.pos.x][y_shift + cell.pos.y] = cell.resource.amount * 255 / 800

        # 15) Self research point
        cur_research_pt = game.state["teamStates"][team]["researchPoints"]
        normal_research = min(cur_research_pt, 200) * 255 / 200
        obs[15] = np.full(self.observation_shape[1:3], fill_value=normal_research, dtype=np.uint8)

        # 16) Opponent research point
        cur_research_pt = game.state["teamStates"][(team + 1) % 2]["researchPoints"]
        normal_research = min(cur_research_pt, 200) * 255 / 200
        obs[16] = np.full(self.observation_shape[1:3], fill_value=normal_research, dtype=np.uint8)

        # 17) Day night cycle ratio
        day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        obs[17] = np.full(self.observation_shape[1:3], fill_value=(game.state["turn"] % day_length) * 255 / day_length, dtype=np.uint8)

        # 18) Current turn number
        obs[18] = np.full(self.observation_shape[1:3],
                          fill_value=game.state["turn"] * 255 / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
                          dtype=np.uint8)

        # 19) Whether is it out of bounds in the map
        obs[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 255

        return obs

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile is not None:
                action = self.actions_cities[action_code % len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
                if isinstance(action, ResearchAction) and game.state["teamStates"][self.team]["researchPoints"] > 200:
                    return None

            else:
                action = self.actions_units[action_code % len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )

            return action
        except Exception as e:
            # Not a valid action
            print(e, file=sys.stderr)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        if action:
            self.match_controller.take_action(action)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        self.researchPoints_last = 0
        self.resource_clusterer = ResourceClusterer(game)

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])
        researchPoints = game.state["teamStates"][self.team]["researchPoints"]


        unit_reward = 0.05;
        rewards = {}
        rewards["rew/r_city_fuel"] = 0
        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            if city.team == self.team:
                city_fuel = city.fuel
                fuel_usage = city.get_light_upkeep()
                fuel_required_full_night = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"] * fuel_usage
                fuel_required_next_night = fuel_required_full_night
                if game.state["turn"] % day_length > GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]:
                    fuel_required_next_night = (day_length - game.state["turn"] % day_length) * fuel_usage
                if city_fuel < fuel_required_next_night:
                    rewards["rew/r_city_fuel"] -= unit_reward * len(city.city_cells) / 100
                elif city_fuel < fuel_required_next_night + fuel_required_full_night:
                    rewards["rew/r_city_fuel"] += unit_reward * len(city.city_cells) / 100
                else:
                    rewards["rew/r_city_fuel"] += 2 * unit_reward * len(city.city_cells) / 100

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1



        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units_created"] = max(unit_count - self.units_last, 0) * unit_reward
        rewards["rew/r_units_lost"] = min(unit_count - self.units_last, 0) * unit_reward * 1.25
        self.units_last = unit_count
        # reward for research
        rewards["rew/r_research"] = (researchPoints - self.researchPoints_last) * unit_reward * (0 if researchPoints > 200 else (0.5 if researchPoints > 50 else 0.75))
        self.researchPoints_last = researchPoints


        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles_created"] = max(city_tile_count - self.city_tiles_last, 0) * unit_reward * 2
        rewards["rew/r_city_tiles_lost"] = min(city_tile_count - self.city_tiles_last, 0) * unit_reward * 2 * 1.5
        self.city_tiles_last = city_tile_count
        '''
        # Reward collecting fuel
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = ((fuel_collected - self.fuel_collected_last) / 20000)
        self.fuel_collected_last = fuel_collected
        '''
        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        rewards["rew/r_clustered_resources"] = self.resource_clusterer.getReward(game, self.team, unit_reward)
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count
            if city_tile_count is 0:
                rewards["rew/r_city_tiles_end"] = -2

            # Curstom reward, reward for winning the game
            # win_team = None
            # city_tile_count = [0, 0]
            # for city in game.cities.values():
            #     city_tile_count[city.team] += len(city.city_cells)
            #
            # if city_tile_count[Constants.TEAM.A] > city_tile_count[Constants.TEAM.B]:
            #     win_team = Constants.TEAM.A
            # elif city_tile_count[Constants.TEAM.A] < city_tile_count[Constants.TEAM.B]:
            #     win_team = Constants.TEAM.B
            #
            # # if tied, count by units
            # unit_count = [
            #     len(game.get_teams_units(Constants.TEAM.A)),
            #     len(game.get_teams_units(Constants.TEAM.B)),
            # ]
            # if unit_count[Constants.TEAM.A] > unit_count[Constants.TEAM.B]:
            #     win_team = Constants.TEAM.A
            # elif unit_count[Constants.TEAM.B] > unit_count[Constants.TEAM.A]:
            #     win_team = Constants.TEAM.B
            #
            # if win_team == self.team:
            #     rewards["rew/r_game_win"] = 10.0  # Win
            # elif win_team is None:
            #     rewards["rew/r_game_win"] = 0  # Tie
            # else:
            #     rewards["rew/r_game_win"] = 0  # Loss
            '''
            # Example of a game win/loss reward instead
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = 100.0 # Win
            else:
                rewards["rew/r_game_win"] = -100.0 # Loss
            '''

        reward = 0
        for name, value in rewards.items():
            reward += value

        return reward

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return
