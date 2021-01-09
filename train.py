from car_game import Car
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K
import pygame
from collections import deque
from time import time
from keras.callbacks import LearningRateScheduler
import psutil
import os

p = psutil.Process(os.getpid())

try:

    p.nice(0)  # set>>> p.nice()10

except:

    p.nice(psutil.HIGH_PRIORITY_CLASS)

# Learning rate decay
def scheduler(epoch, lr):
    lr *=1-9e-7
    return lr


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.9 # Weight of past data
        self.epsilon = 1.0 # Balance between random and predicted actions
        self.epsilon_min = 0.01 # Minimum epsilon
        self.epsilon_decay = 0.995 # Rate of decay
        self.learning_rate = 0.001 # Network learning rate
        self.model = self.build_model()


    def build_model(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def remember(self, state, action, reward, new_state):
        self.memory.append([state, action, reward, new_state])


    def act(self, state):

        # Each interaction reduces epsilon, until it gets to epsilon min
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # If random n < epsilon, random action, else predicted action
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Predicted action
        return np.argmax(self.model.predict(state)[0])


    def replay(self, batch_size):

        # If memory list not big enough do nothing
        if len(self.memory) < batch_size:   return

        # Randomly selects samples to train
        samples = random.sample(self.memory, batch_size)

        # Iterates between all sampled train data
        for sample in samples:

            state, action, reward, new_state  = sample
            target = self.model.predict(state)
            Q_future = max(self.model.predict(new_state)[0])
            target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0, callbacks=[LearningRateScheduler(scheduler)])


    def save_model(self, fn):
        self.model.save(fn)


if __name__ == "__main__":

    env = Car(True, True, True, True, True) # Random start position, mirroring, showing reward
    agent = DQNAgent(env.state_size, env.action_size)
    trials = 100

    # First action
    _, _, _, state = env.run('ACCEL')

    for step in range(trials):

        for trial in range(512):

            t1 = time()

            # Choose an action according to the current state
            action = agent.act(state)

            # Do the action on the environment and get returning info
            _, reward, env_action, next_state = env.run(action)

            # Store info to train the network
            agent.remember(state, action, reward, next_state)

            state = next_state

            agent.replay(32)

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_q]:
                pygame.quit()
                break

            # Gets model learning rate
            lr = K.eval(agent.model.optimizer.lr)

            print('Step:{}  Trial:{}  lr: {} Speed:{}  Reward:{}  Time: {}  Action:{}'.format(step, trial, lr, env.speed, round(reward,2), round(time() - t1, 2), env_action))

            # Saves with frequency f
            if trial % 64 == 0:

                agent.save_model("./models/temp_success.h5")

    agent.save_model("./models/success.h5")

