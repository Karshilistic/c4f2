from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

from gen_sim import gen_sim

COMPETITION_ROUND = 1  # 1 or 2, depending on which competition round you are in
YELLOW_TIME = 6
HOLD_TIME = 6
MAX_STEP_COUNT = 1000
CENSOR_PROBABILITY = 0.1

class SimulationEnv(py_environment.PyEnvironment):

    def __init__(self, competition_round):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,9), dtype=np.int32, name='observation')

        print('Starting Sumo...')
        # The normal way to start sumo on the CLI
        self._sumoBinary = checkBinary('sumo')
        # comment the line above and uncomment the following one to instantiate the simulation with the GUI
        # sumoBinary = checkBinary('sumo-gui')

        traci.switch("contestant")
        traci.close()

        # Generate an episode with the specified probabilities for lanes in the intersection
        # Returns the number of vehicles that will be generated in the episode
        self._vehicles = gen_sim('', round=competition_round,
                           p_west_east=0.5, p_east_west=0.2,
                           p_north_south=0.2, p_south_north=0.1)

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([self._sumoBinary, "-c", "data/cross.sumocfg",
                     "--time-to-teleport", "-1",
                     "--tripinfo-output", "tripinfo.xml", '--start', '-Q'], label='contestant')
        # Connection to simulation environment
        self._conn = traci.getConnection("contestant")

        self._competition_round = competition_round
        # self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._episode_ended = False
        self._time_step = 0
        # we start with phase 2 where EW has green
        self._conn.trafficlight.setPhase("0", 2)
        self._total_waiting_time = 0
        self._total_emissions = 0
        self._waiting_time_per_episode = []
        self._state = get_state(self._conn, self._competition_round)
        self._waiting_times = []
        self._time_step = 0
        self._gamma = 1.0
        self._different_action = 0
        self._last_action = 0
        self.number_of_cars = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        traci.switch("contestant")
        traci.close()

        # Calculate the avg waiting time per vehicle
        avg_waiting_time = self._total_waiting_time / self._vehicles
        avg_emissions = self._total_emissions / (1000 * self._vehicles)
        self._waiting_time_per_episode.append(avg_waiting_time)
        print('Average waiting time = ' + str(avg_waiting_time)
              + ' (s) -- Average Emissions (CO2) = ' + str(avg_emissions) + "(g)")
        
        self._total_waiting_time = 0
        self._total_emissions = 0
        self.number_of_cars = 0
        self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._episode_ended = False
        self._time_step = 0
        self._vehicles = gen_sim('', round=self._competition_round,
                           p_west_east=0.5, p_east_west=0.2,
                           p_north_south=0.2, p_south_north=0.1)

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([self._sumoBinary, "-c", "data/cross.sumocfg",
                     "--time-to-teleport", "-1",
                     "--tripinfo-output", "tripinfo.xml", '--start', '-Q'], label='contestant')
        # Connection to simulation environment
        self._conn = traci.getConnection("contestant")

        return ts.restart(np.array([self._state], dtype=np.int32))
    
    def _step(self, action):
        # print('Step ', self._time_step)
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        # Make sure episodes don't go on forever.
        if self._conn.simulation.getMinExpectedNumber() <= 0 or self._time_step >= MAX_STEP_COUNT:
            number_of_cars = get_moving_count(self._conn, self._vehicle_ids) + get_waiting_count(self._conn, self._vehicle_ids)
            self._conn.close()
            self._episode_ended = True
        elif self._conn.simulation.getMinExpectedNumber() > 0 and self._time_step <= MAX_STEP_COUNT:
            self._vehicle_ids = self._conn.lane.getLastStepVehicleIDs("1i_0") \
                        + self._conn.lane.getLastStepVehicleIDs("2i_0") \
                        + self._conn.lane.getLastStepVehicleIDs("4i_0") \
                        + self._conn.lane.getLastStepVehicleIDs("3i_0")
            # if agent is not None:
            #     if train:
            #         action = agent.select_action(state, conn, vehicle_ids)
            #     else:
            #         action = agent.select_action(state)
            if action not in range(0, 2):
                print("Agent returned an invalid action")
            cur_waiting_time, elapsed, emissions = take_action(self._conn, self._state, action, self._competition_round)
            next_state = get_state(self._conn, self._competition_round)
            self._state = next_state
            # else:
            #     cur_waiting_time = get_waiting_count(conn, vehicle_ids)
            #     emissions = get_total_co2(conn, vehicle_ids)
            #     elapsed = 1
            #     conn.simulationStep()
            #     next_state = get_state(conn, competition_round)
            #     state = next_state
            self._total_waiting_time += cur_waiting_time
            self._total_emissions += emissions
            self._waiting_times.append(cur_waiting_time)
            self._time_step += elapsed
            number_of_cars = get_moving_count(self._conn, self._vehicle_ids) + get_waiting_count(self._conn, self._vehicle_ids)
        else:
            raise ValueError('samsing wrong.')

        if self._last_action == action:
            self._different_action = 0
        else:  
            self._different_action = 1
        self._last_action = action

        if number_of_cars > 0:
            reward = ( -0.2 * self._different_action ) - ( 0.8 * self._total_waiting_time / number_of_cars )
        else:
            reward = -self._different_action

        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=reward, discount=self._gamma)

    def stop():
        traci.switch("contestant")
        traci.close()


def get_curr_open_dir(conn):
    """

    :param conn: The simulation connection environment
    :return: curr_open_dir

    - curr_open_dir for COMPETITION_ROUND 1:
            (0 for vertical, 1 for horizontal) --> possible actions (0, 1)
    """
    return [conn.trafficlight.getPhase("0") // 2]


def get_waiting_count(conn, vehicle_ids):
    """

    :param conn: The simulation connection environment
    :param vehicle_ids: Vehicle ids in the simulation
    :return: the number of waiting vehicles.

    We define a waiting vehicle by passing its ID to the getWaitingTime SUMO function
    incrementing the waiting vehicles count by 1 whenever a car’s waiting time is more than 0.

    It uses the function getWaitingTime(self, vehID)
    which returns the waiting time of a vehicle, defined as the time (in seconds) spent with
    a speed below 0.1m/s since the last time it was faster than 0.1m/s.

    (basically, the waiting time of a vehicle is reset to 0 every time it moves).

    A vehicle that is stopping intentionally with a <stop> does not accumulate waiting time.

    """
    res = 0
    for vehicle_id in vehicle_ids:
        if conn.vehicle.getWaitingTime(vehicle_id) > 0:
            res += 1
    return res


def get_total_waiting_time(conn, vehicle_ids):
    """

    :param conn: The simulation connection environment
    :param vehicle_ids: Vehicle ids in the simulation
    :return:    a number representing the total waiting times of
                all vehicles whose ids are in the vehicle_ids list.

    It uses the function getWaitingTime described earlier
    to accumulate all waiting times of vehicles.
    """
    res = 0
    for vehicle_id in vehicle_ids:
        if conn.vehicle.getWaitingTime(vehicle_id) > 0:
            res += conn.vehicle.getWaitingTime(vehicle_id)
    return res


def get_total_co2(conn, vehicle_ids):
    """

    :param conn: The simulation connection environment
    :param vehicle_ids: Vehicle ids in the simulation
    :return:    a number representing the total CO2 emissions (in the last step)
                from all vehicles whose ids are in the vehicle_ids list.
                units: mg
    """
    res = 0
    for vehicle_id in vehicle_ids:
        res += conn.vehicle.getCOEmission(vehicle_id)
    return res


def get_total_accumulated_waiting_time(conn, vehicle_ids):
    """

    :param conn: The simulation connection environment
    :param vehicle_ids: Vehicle ids in the simulation
    :return: the sum of the accumulated waiting times of
            vehicles since they entered the simulation.

    (Note: this is different from getWaitingTime since the latter
    resets the waiting time of a car to 0 if it moves with a speed
    faster than 0.1m/s)
    """
    res = 0
    for vehicle_id in vehicle_ids:
        if conn.vehicle.getAccumulatedWaitingTime(vehicle_id) > 0:
            res += conn.vehicle.getAccumulatedWaitingTime(vehicle_id)
    return res


def get_total_speed(conn, vehicle_ids):
    """

    :param conn: The simulation connection environment
    :param vehicle_ids: Vehicle ids in the simulation
    :return: the sum of the current speeds of all cars in the simulation

    This function makes use the getSpeed(vehicle_id) which returns the speed
    in m/s of a single vehicle in the simulation within the last step (second).
    """
    res = 0
    for vehicle_id in vehicle_ids:
        res += conn.vehicle.getSpeed(vehicle_id)
    return res


# how many cars are moving (whose speed is higher than threshold making waiting time = 0)
def get_moving_count(conn, vehicle_ids):
    """

    :param conn: The simulation connection environment
    :param vehicle_ids: Vehicle ids in the simulation
    :return: the number of cars that are not stationary
            (i.e. moving with a speed larger than 0.1 m/s)

    It uses of the getWaitingTime(vehicle_id) function defined earlier.
    """
    res = 0
    for vehicle_id in vehicle_ids:
        if conn.vehicle.getWaitingTime(vehicle_id) == 0:
            res += 1
    return res


def get_state(conn, competition_round=1):
    """

    :param competition_round: current round of the competition
    :param conn: The simulation connection environment
    :return: the current state of the simulation as defined previously.

    The current state is defined as:

    state = [curr_open_dir, 8*detector(waiting times)]
    Where:
    - detector[i]: Waiting time for the vehicle on detector[i]
                    since it was last moving with speed > 0.1 m/s.
    - detector[i] for i in [0-3] is near traffic light
    - detector[i] for i in [4-7] is far from traffic light
    - For illustration of detector positions and numbering (check attached sensor_data.png)

    - curr_open_dir for COMPETITION_ROUND 1:
            (0 for vertical, 1 for horizontal) --> possible actions (0, 1)
    """

    result = get_curr_open_dir(conn)  # current direction
    # 8: Number of sensors
    for i in range(8):
        # get if there is a vehicle in front of the detector i
        if len(conn.inductionloop.getVehicleData(str(i))) > 0:
            # get vehicle id
            vehicle_id = conn.inductionloop.getVehicleData(str(i))[0][0]
            # get waiting time of a vehicle and append in result
            result.append(conn.vehicle.getWaitingTime(vehicle_id))
        else:
            # No vehicle --> 0 waiting time
            result.append(0)

    if competition_round == 2:
        print("ROUND 2 yet to be released!!")
        raise NotImplementedError

    return result


def take_action(conn, state, action, competition_round):
    """

    :param conn: The simulation connection environment
    :param state: state of the simulation as defined previously
    :param action: integer denoting the action taken by the agent.
            - actions for COMPETITION_ROUND 1: (0 vertical, 1 horizontal)
    :param competition_round: the current competition round (1 or 2)
    :return: (waiting, elapsed) where:
            waiting: total waiting time of all vehicles during this action
            elapsed: the time steps elapsed since this action (YELLOW_TIME+HOLD_TIME)


    """

    curr_action = state[0]
    # HOLD_TIME = the minimum time between traffic color switches.
    # A green light cannot switch to yellow unless HOLD_TIME has passed.
    # By default, HOLD_TIME = 6 seconds
    elapsed = HOLD_TIME

    # waiting: total waiting time of all vehicles during this action
    waiting = 0
    emissions = 0
    # vehicles in the simulation during this action
    vehicle_ids = conn.lane.getLastStepVehicleIDs("1i_0") \
                + conn.lane.getLastStepVehicleIDs("2i_0") \
                + conn.lane.getLastStepVehicleIDs("4i_0") \
                + conn.lane.getLastStepVehicleIDs("3i_0")
    # if chosen direction is different from the current direction, switch
    if int(action) != int(curr_action):

        currentPhase = int(conn.trafficlight.getPhase("0"))
        """
        currentPhase = trafficlight phase from the connection (including the yellow)
        - currentPhase for COMPETITION_ROUND 1: (0 for vertical, 2 for horizontal)
        odd currentPhase numbers are for the yellow phase of the traffic light.
        The simulation only updates the user on the open directions (even numbers)
        Possible actions (curr_open_dir) = currentPhase//2
        """
        # Switch to yellow
        nxt = (currentPhase + 1) % 4
        conn.trafficlight.setPhase("0", nxt)

        # Add switching time to the elapsed time
        elapsed += YELLOW_TIME

        # commit to switching time keeping track of waiting time
        for i in range(YELLOW_TIME):
            conn.simulationStep()  # increment the time in simulation
            waiting += get_waiting_count(conn, vehicle_ids)
            emissions += get_total_co2(conn, vehicle_ids)
        # commit to hold time

    # Hold for HOLD_TIME, keeping track of waiting time
    for i in range(HOLD_TIME):
        conn.simulationStep()
        waiting += get_waiting_count(conn, vehicle_ids)
        emissions += get_total_co2(conn, vehicle_ids)
    return waiting, elapsed, emissions


# client testing framework
def run_episode(conn, agent, competition_round, train=True):
    """

    :param conn: The simulation connection environment
    :param agent: The action-taking agent object
    :param competition_round: the current competition round (1 or 2)
    :param train: train or test flag
            train is True:
                Agent's select_action method has access to 'vehicle_ids'
                vehicle_ids: a list of vehicles still in the simulation currently
            train is False:
                Agent's select_action simulates the actual testing environment
    :return: (total_waiting_time, waiting_times)
            total_waiting_time: the sum of the waiting times of all vehicles that ever existed
                                in the simulation episode
            waiting_times: a list of total_waiting_times per action
                            len(waiting_times) = # of actions taken
                            by the agent in the current episode
    """
    step = 0
    # we start with phase 2 where EW has green
    conn.trafficlight.setPhase("0", 2)
    total_waiting_time = 0
    total_emissions = 0
    state = get_state(conn, competition_round)
    waiting_times = []
    # Start simulation
    while conn.simulation.getMinExpectedNumber() > 0 and step <= MAX_STEP_COUNT:
        vehicle_ids = conn.lane.getLastStepVehicleIDs("1i_0") \
                    + conn.lane.getLastStepVehicleIDs("2i_0") \
                    + conn.lane.getLastStepVehicleIDs("4i_0") \
                    + conn.lane.getLastStepVehicleIDs("3i_0")
        if agent is not None:
            if train:
                action = agent.select_action(state, conn, vehicle_ids)
            else:
                action = agent.select_action(state)
            if action not in range(0, 2):
                print("Agent returned an invalid action")
            cur_waiting_time, elapsed, emissions = take_action(conn, state, action, competition_round)
            next_state = get_state(conn, competition_round)
            state = next_state
        else:
            cur_waiting_time = get_waiting_count(conn, vehicle_ids)
            emissions = get_total_co2(conn, vehicle_ids)
            elapsed = 1
            conn.simulationStep()
            next_state = get_state(conn, competition_round)
            state = next_state
        total_waiting_time += cur_waiting_time
        total_emissions += emissions
        waiting_times.append(cur_waiting_time)
        step += elapsed
    conn.close()
    return total_waiting_time, waiting_times, total_emissions

train_env = SimulationEnv(1)
eval_env = SimulationEnv(1)
# utils.validate_py_environment(environment, episodes=5)
tf_train_env = tf_py_environment.TFPyEnvironment(train_env)
tf_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

time_step = tf_env.reset()
rewards = []
steps = []
number_of_episodes = 200000
COLLECTION_STEPS = 1
BATCH_SIZE = 64
EVAL_EPISODES = 10
EVAL_INTERVAL = 1000

for _ in range(number_of_episodes):
    episode_reward = 0
    episode_steps = 0
    while not time_step.is_last():
        action = tf.random.uniform([1], 0, 2, dtype=tf.int32)
        time_step = tf_env.step(action)
        episode_steps += 1
        episode_reward += time_step.reward.numpy()
        print("Action: ", action, "     Reward: ", episode_reward)
    rewards.append(episode_reward)
    steps.append(episode_steps)
    time_step = tf_env.reset()

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

SimulationEnv.stop()