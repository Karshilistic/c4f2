import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
import random
import numpy as np

from Agent import Agent
from sumo_utils import run_episode, get_state
from gen_sim import gen_sim

NUM_EPISODES = 30  # Number of complete simulation runs
COMPETITION_ROUND = 1  # 1 or 2, depending on which competition round you are in
random.seed(COMPETITION_ROUND)

# Hyperparameters
NUMBER_OF_LAYERS = 1    # Number of layers of NN
LAYER_WIDTH = 30         # Number of neurons in layer
LEARNING_RATE = 0.0001   # Alpha, learning rate
GAMMA = 0.75            # Discount Factor
TRAINING_EPOCHS = 100
BATCH_SIZE = 32

NUM_OF_STATES = 9       # 1 current state, 8 detectors
NUM_OF_ACTIONS = 1      # either 0 or 1

"""
state = [curr_open_dir, 8*detector(waiting times)]
Where:
- detector[i]: Waiting time for the vehicle on detector[i] since it was last moving with speed > 0.1 ms^{-1}
- detector[i] for i in [0-3] is near traffic light
- detector[i] for i in [4-7] is far from traffic light 
- For illustration of detector positions and numbering (check attached sensor_data.png)
----------------------------------------------------------------------------------------
- curr_open_dir for COMPETITION_ROUND 1: (0 for vertical, 1 for horizontal) --> possible actions (0, 1)
"""

if __name__ == "__main__":

    print('Starting Sumo...')
    # The normal way to start sumo on the CLI
    sumoBinary = checkBinary('sumo')
    # comment the line above and uncomment the following one to instantiate the simulation with the GUI

    # sumoBinary = checkBinary('sumo-gui')

    agent = Agent(NUMBER_OF_LAYERS, LAYER_WIDTH, LEARNING_RATE, GAMMA, TRAINING_EPOCHS, NUM_EPISODES)  # Instantiate your agent object
    waiting_time_per_episode = []  # A list to hold the average waiting time per vehicle returned from every episode
    
    # epsilon = 0                     # Used in Epsilon-greedy policy (increases exploration)

    for e in range(NUM_EPISODES):
        # Generate an episode with the specified probabilities for lanes in the intersection
        # Returns the number of vehicles that will be generated in the episode
        vehicles = gen_sim('', round=COMPETITION_ROUND,
                           p_west_east=0.3, p_east_west=0.2,
                           p_north_south=0.2, p_south_north=0.1)

        print('Starting Episode ' + str(e) + '...')

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                     "--time-to-teleport", "-1",
                     "--tripinfo-output", "tripinfo.xml", '--start', '-Q'], label='contestant')
        # Connection to simulation environment
        conn = traci.getConnection("contestant")
        # Run a complete simulation episode with the agent taking actions for as long as the episode lasts.
        # An episode lasts as long as there are cars in the simulation AND the time passed < 1000 seconds
        total_waiting_time, waiting_times, total_emissions = run_episode(conn, agent, COMPETITION_ROUND)
        # Cleaning up TraCi environments
        traci.switch("contestant")
        traci.close()
        # Calculate the avg waiting time per vehicle
        avg_waiting_time = total_waiting_time / vehicles
        avg_emissions = total_emissions / (1000 * vehicles)
        waiting_time_per_episode.append(avg_waiting_time)

        print('episode[' + str(e) + '] Average waiting time = ' + str(avg_waiting_time)
              + ' (s) -- Average Emissions (CO2) = ' + str(avg_emissions) + "(g)")
        

        for _ in range(TRAINING_EPOCHS):
            # REPLAY FUNCTION
            sample = agent.get_memory()
            ## Get random samples of defined batch size
            if len(sample) < BATCH_SIZE:
                batch = random.sample(sample, len(sample))
            else:
                batch = random.sample(sample, BATCH_SIZE)

            ## Get states
            # if len(batch) > 0:
            state = []
            next_state = []
            for elem in range(len(batch)):
                state.append(batch[elem][0])
                next_state.append(batch[elem][3])

            state = np.array(state)
            next_state = np.array(next_state)

            ## Get predicted state
            q_sa = agent.predict(state)
            q_sa_future = agent.predict(next_state)
            # print(q_sa, " ", q_sa_future)
            ## Train
            ### Prepare training data
            X = np.zeros((len(batch), NUM_OF_STATES))
            y = np.zeros((len(batch), NUM_OF_ACTIONS))

            for i, elem in enumerate(batch):
                # print(i, " ", elem)
                state_, action_, reward_= elem[0], elem[1], elem[2]
                # print(state_)
                # print(action_)
                # print(reward_)
                q_sa_current = q_sa[i]
                # print(q_sa_current)
                q_sa_current = reward_ + GAMMA * np.amax(q_sa_future[i])
                # print(q_sa_current)
                X[i] = state_
                y[i] = q_sa_current
            # print(X)
            # print(y)
            agent.train_model(X, y)

