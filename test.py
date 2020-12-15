from car_game import Car
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pygame
from time import time
import os
import psutil

p = psutil.Process(os.getpid())

try:

    p.nice(0)  # set>>> p.nice()10

except:

    p.nice(psutil.HIGH_PRIORITY_CLASS)

clock = pygame.time.Clock()

path = os.getcwd()

# Main class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.load_trained_model()


    def build_model(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        return model


    def act(self, state):
        return np.argmax(self.model.predict(state)[0])


    def load_trained_model(self):
       model = self.build_model()
       model.load_weights(path+"./models/success.h5")
       return model


if __name__ == "__main__":

    env = Car(show_proximity=True)
    agent = DQNAgent(env.state_size, env.action_size)
    _, _, _, state = env.run('ACCEL')

    while True:

        t1 = time()
        action = agent.act(state)
        _, reward, env_action, next_state  = env.run(action)
        state = next_state

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_q]:
            pygame.quit()
            break

        clock.tick(30)

        print('Reward:{}  Time: {}  Action:{}'.format(round(reward,2), round(time() - t1, 2), env_action))



