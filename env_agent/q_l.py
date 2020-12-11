from collections import defaultdict
from env_agent.agent import Agent
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import water_envs

env = gym.make('water_envs-v0')
interval = 10
eps = 0.1
ep_c = 10000
# for epsilon
EPSILON_DECAY_LAST_FRAME = 10
EPSILON_START = 0.5
EPSILON_FINAL = 0.01

class QLearningAgent(Agent):

    def __init__(self):
        super().__init__()
        self.frame_idx = 0
        self.ts_frame = 0

    def learn(self, episode_count=ep_c, gamma=0.9,
              learning_rate=0.1, render=False):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        print('self.Q:{}'.format(self.Q))
        reward_list = [np.inf]
        eps_list = []
        for e in range(episode_count):
            env.reset()
            s = env.point_finder()
            done = False
            self.frame_idx += 1
            epsilon = self.epsilon_selection(e, self.frame_idx, mode='fluctuation')
            eps_list.append(epsilon)
            while not done:
                if render:
                    env.render()
                a = self.policy(s, actions, epsilon)
                trace, state, reward, obs, done = env.step(a)
                gain = reward + gamma * max(self.Q[state])
                estimated = self.Q[s][a]
                #td_error = gain - estimated
                self.Q[s][a] += learning_rate * (gain - estimated)#td_error
                s = state
                # to show the trace having the best reward
                reward_list.append(reward)
                if reward_list[-1] >= reward_list[-2]:
                    trace_list = trace
            else:
                self.log(reward)
            if e != 0 and e % interval == 0:
                self.show_reward_log(episode=e)
        print(trace_list)
        return trace_list, obs, eps_list

    def show_reward_log(self, interval=interval, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            # to compute mean
            mean = np.round(np.mean(rewards), 3)
            # to compute std
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{})".format(episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:i + interval]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            return indices, means, stds

    def epsilon_selection(self, e, idx, mode='constant'):
        if mode == 'constant':
            epsilon = 0.1
        if mode == 'fluctuation':
            idx += 1
            # the timing of turing to 0.01
            # 1.0e-4: 220
            # 1.0e-5:990
            # 1.0e-6:3300
            # 1.0e-6 * 0.2:7000
            # 1.0e-7:9900
            speed = (idx - self.ts_frame) * (e + 1) * 1.0e-6
            epsilon = max(EPSILON_FINAL, EPSILON_START - idx / EPSILON_DECAY_LAST_FRAME * (speed))
            self.ts_frame = idx
        else:
            Exception('Chose mode(constant or fluctuation)')

        return epsilon


class SARSAAgent(Agent):

    def __init__(self):
        super().__init__()
        self.frame_idx = 0
        self.ts_frame = 0

    def learn(self, episode_count=ep_c, gamma=0.9,
              learning_rate=0.1, render=False):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        reward_list = [np.inf]
        eps_list = []
        for e in range(episode_count):
            env.reset()
            s = env.point_finder()
            done = False
            self.frame_idx += 1
            epsilon = self.epsilon_selection(e, self.frame_idx, mode='constant')
            eps_list.append(epsilon)
            a = self.policy(s, actions, epsilon)
            while not done:
                if render:
                    env.render()
                trace, n_state, reward, obs, done = env.step(a)
                n_action = self.policy(n_state, actions, epsilon)
                gain = reward + gamma * self.Q[n_state][n_action]
                estimated = self.Q[s][a]
                td_error = gain - estimated
                self.Q[s][a] += learning_rate * td_error
                s = n_state
                a = n_action
                # to show the trace having the best reward
                reward_list.append(reward)
                if reward_list[-1] >= reward_list[-2]:
                    trace_list = trace
            else:
                self.log(reward)
            if e != 0 and e % interval == 0:
                self.show_reward_log(episode=e)
        print(trace_list)
        return trace_list, obs, eps_list

    def show_reward_log(self, interval=interval, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            # to compute mean
            mean = np.round(np.mean(rewards), 3)
            # to compute std
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{})".format(episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:i + interval]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            return indices, means, stds

    def epsilon_selection(self, e, idx, mode='constant'):
        if mode == 'constant':
            epsilon = 0.1
        if mode == 'fluctuation':
            idx += 1
            speed = (idx - self.ts_frame) * (e + 1) * 1.0e-6
            epsilon = max(EPSILON_FINAL, EPSILON_START - idx / EPSILON_DECAY_LAST_FRAME * (speed))
            self.ts_frame = idx
        else:
            Exception('Chose mode(constant or fluctuation)')

        return epsilon

def show(ax, trace, obs):
    ims = []
    nrows, ncols = env.MAP_shape
    fig = plt.figure()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(obs)
    for row, col in trace:
        canvas[(row, col)] = 0
        img1 = plt.imshow(canvas, interpolation="none", cmap=cm.GnBu)
        ims.append([img1])
    img = plt.imshow(canvas, interpolation="none", cmap=cm.GnBu, animated=True)
    #ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    return img


def train(method):
    if method == 'ql':
        agent = QLearningAgent()
    if method == 'sarsa':
        agent = SARSAAgent()

    trace, obs, epsilon = agent.learn()
    indices, means, stds = agent.show_reward_log()

    std_p = [x + y for (x, y) in zip(means, stds)]
    std_m = [x - y for (x, y) in zip(means, stds)]
    print('reward means:{}, std:{}'.format(np.mean(means[-100:]), np.mean(stds[-100:])))
    # for trace
    ax = plt.gca()
    ani = show(ax, trace, obs)
    # for plotting
    #plt.figure(figsize=(5, 3), dpi=150)
    if method == 'ql':
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title("Q-learning")
        #plt.ylim(-40, 20)
        ax1.grid()
        ax1.fill_between(indices, std_m, std_p, alpha=0.1, color='b', label="QL:Range from MEAN+STD to MEAN-STD)")
        ax1.plot(indices, means, alpha=0.8, color='b', label="QL:Rewards for each {} episode(alpha=0.01)".format(interval))
        ax1.set_xlabel('episode')
        ax1.set_ylabel('reward')
        # plt.tick_params(labelbottom=False)
        ax1.legend(loc='lower right', borderaxespad=1, fontsize=5)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title("Transition of Epsilon")
        # plt.ylim(-40, 20)
        ax2.grid()
        ax2.plot([i for i in range(len(epsilon))], epsilon, color='black',
                 label="epsilon")
        ax2.set_xlabel('episode')
        ax2.set_ylabel('epsilon')
        # plt.tick_params(labelbottom=False)
        ax2.legend(loc='lower right', borderaxespad=1, fontsize=5)
        plt.subplots_adjust(hspace=0.4)
        plt.show()

    if method == 'sarsa':
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title("SARSA")
        # plt.ylim(-40, 20)
        ax1.grid()
        ax1.fill_between(indices, std_m, std_p, alpha=0.1, color='b', label="SARSA:Range from MEAN+STD to MEAN-STD)")
        ax1.plot(indices, means, alpha=0.8, color='b',
                 label="SARSA:Rewards for each {} episode(alpha=0.01)".format(interval))
        ax1.set_xlabel('episode')
        ax1.set_ylabel('reward')
        # plt.tick_params(labelbottom=False)
        ax1.legend(loc='lower right', borderaxespad=1, fontsize=5)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title("Transition of Epsilon")
        # plt.ylim(-40, 20)
        ax2.grid()
        ax2.plot([i for i in range(len(epsilon))], epsilon, color='black',
                 label="epsilon")
        ax2.set_xlabel('episode')
        ax2.set_ylabel('epsilon')
        # plt.tick_params(labelbottom=False)
        ax2.legend(loc='lower right', borderaxespad=1, fontsize=5)
        plt.subplots_adjust(hspace=0.4)
        plt.show()


if __name__ == '__main__':
    train(method='ql')