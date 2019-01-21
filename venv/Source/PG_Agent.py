import matplotlib

matplotlib.use('TkAgg')

import copy
import pylab
import random
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 1000

class PolicyGradient_Agent:
    def __init__(self, state_size):
        self.load_model = False
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.state_size = state_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/PolicyGradient_Agent.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    def optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Cross Entropy
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)
        return train

    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    def train_model(self):
        discounted_rewards = np.float32(self.discounted_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    env = Env()
    state_size = env.state_size
    print("State_size: ", state_size)
    agent = PolicyGradient_Agent(state_size)

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        goal = 0

        while not done:
            global_step += 1
            action = agent.get_action(state)

            next_state, reward, done, goal = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)
            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # Updates Policy Network
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                done = "Crash"

                # if goal == 1:
                #     goal = "Goal!"
                # else:
                #     goal = "No Goal…"

                # by Episode
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/PolicyGradient_Agent.png")
                print("episode:", e, "  score:", score, "global_step: ",
                      global_step, "Result : ", done)

        # Save Weight
        if e % 100 == 0:
            agent.model.save_weights("./save_model/PolicyGradient_Agent.h5")