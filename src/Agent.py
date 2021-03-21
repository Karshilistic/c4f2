import random
import tensorflow as tf
import numpy as np
import sumo_utils

class Agent:
    def __init__(self, num_layers, layer_width, learn_rate, gamma, epochs, maxEpisodes):
        self._num_layers = num_layers
        self._layer_width = layer_width
        self._learn_rate = learn_rate
        self._gamma = gamma
        self._epochs = epochs
        self._maxEpisodes = maxEpisodes
        
        inputs = tf.keras.Input(shape=(9,))
        x = tf.keras.layers.Dense(self._layer_width, activation = 'relu')(inputs)
        outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
        model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'test1')
        optimizer = tf.keras.optimizers.Adam(learning_rate = self._learn_rate)
        model.compile(optimizer = optimizer, loss = 'mean_squared_error' )
        
        self._model = model

        self._current_accumulated_waiting_time = 0
        self._old_accumulated_waiting_time = 0
        self._old_state = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self._old_action = -1
        self._memory = []
        self._epsilon = 0.2

    def select_action(self, state, conn=None, vehicle_ids=None):
        current_state = state
        # Calculate reward; reward is higher the smaller the wait time difference is, i.e, decreased the wait time (with a max of 0)
        print(sumo_utils.get_total_accumulated_waiting_time(conn, vehicle_ids))
        self._current_accumulated_waiting_time += sumo_utils.get_total_accumulated_waiting_time(conn, vehicle_ids) #total_waiting_time
        # print(current_total_wait)
        reward = self._old_accumulated_waiting_time - self._current_accumulated_waiting_time
        self._old_accumulated_waiting_time = self._current_accumulated_waiting_time
        print(reward)
            # Prep sample
        sample = (self._old_state, self._old_action, reward, current_state)
            ## save sample
        self._memory.append(sample)

        ## Prepare for next sample
        self._old_state = current_state
        
        if random.random() < self._epsilon:
            action = random.randint(0, 1)
            self._old_action = action
            # print("Action: ", self._old_action)
            return self._old_action
        else:
            # print("using model")
            current_state = np.reshape(current_state, (1, 9))
            self._old_action = int(self._model.predict(current_state) > 0.5)
            # print("Action: ", self._old_action)
            return self._old_action

    def train_model(self, state, q_sa):
        self._model.fit(state, q_sa, epochs=1, verbose=1)

    def predict(self, states):
        return self._model.predict(states)

    def save_model(self, path):
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes = True, show_layer_names = True)

    def get_memory(self):
        return self._memory
