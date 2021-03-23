import random
import tensorflow as tf
import numpy as np
import sumo_utils
from collections import deque

class Agent:
    def __init__(self, num_layers, layer_width, learn_rate, gamma, epochs, maxEpisodes, batch_size):
        self._num_layers = num_layers
        self._layer_width = layer_width
        self._learn_rate = learn_rate
        self._gamma = gamma
        self._epochs = epochs
        self._maxEpisodes = maxEpisodes
        self._batch_size = batch_size
        
        inputs = tf.keras.Input(shape=[9])
        x = tf.keras.layers.Dense(self._layer_width, activation = 'relu')(inputs)
        for _ in range(self._num_layers):
            x = tf.keras.layers.Dense(self._layer_width, activation = 'relu')(x)
        outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

        model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'test1')
        optimizer = tf.keras.optimizers.Adam(learning_rate = self._learn_rate)
        model.compile(optimizer = optimizer, loss = 'mean_squared_error' )
        model.summary()
        self._model = model
        
        self._current_accumulated_waiting_time = 0
        self._old_accumulated_waiting_time = 0
        self._old_state = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self._old_action = -1
        self._memory = []
        self._epsilon = 0.4
        self._replay_buffer = deque(maxlen=2000)

    def sample_experience(self, batch_size):
        indices = np.random.randint(len(self._replay_buffer), size=batch_size)
        batch = [self._replay_buffer[index] for index in indices]
        states, actions, rewards, next_states = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(4)]
        return states, actions, rewards, next_states

    def select_action(self, state, conn=None, vehicle_ids=None):
        current_state = state
        ## Calculate reward; reward is higher the smaller the wait time difference is, i.e, decreased the wait time (with a max of 0)
        self._current_accumulated_waiting_time += sumo_utils.get_total_accumulated_waiting_time(conn, vehicle_ids) #total_waiting_time
        # print(current_total_wait)
        reward = self._old_accumulated_waiting_time - self._current_accumulated_waiting_time
        self._old_accumulated_waiting_time = self._current_accumulated_waiting_time
        print(sumo_utils.get_total_accumulated_waiting_time(conn, vehicle_ids), "   ", self._current_accumulated_waiting_time, "    ", reward)
        ## Prep sample
        sample = (self._old_state, self._old_action, reward, current_state)
        ## save sample
        # self._memory.append(sample)
        self._replay_buffer.append(sample)
        ## Prepare for next sample
        self._old_state = current_state
        
        if random.random() < self._epsilon:
            action = random.randint(0, 1)
            self._old_action = action
            print("Random Action: ", self._old_action)
            return self._old_action
        else:
            # print("using model")
            current_state = np.reshape(current_state, (1, 9))
            self._old_action = int(self._model.predict(current_state) > 0.5)
            print("Action: ", self._old_action)
            return self._old_action

    def train_model(self, state, q_sa):
        self._model.fit(state, q_sa, epochs=1, verbose=1)

    def predict(self, states):
        return self._model.predict(states)

    def reset(self):
        self._current_accumulated_waiting_time = 0
        self._old_accumulated_waiting_time = 0

    def save_model(self, path):
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes = True, show_layer_names = True)

    def get_memory(self):
        # return self._memory
        return self._replay_buffer
