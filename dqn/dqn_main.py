import torch as T
import gym
import numpy as np
from dqn.utils import plotLearning, show_maze
import datetime
import water_envs

mode = 'non_cnn'
if mode == 'cnn':
    from dqn.agent_cnn import Agent
elif mode == 'non_cnn':
    from dqn.agent_no_cnn import Agent
else:
    Exception('plz select cnn or non_cnn')


def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


def comp_time(time):
    if 'seconds' in time:
        s = time.split('.').tolist()
        return s[0] + '_' + s[1] + 'sec'
    elif 'minutes' in time:
        s = time.split('.').tolist()
        return s[0] + '_' + s[1] + 'min'
    else:
        s = time.split('.').tolist()
        return s[0] + '_' + s[1] + 'hr'


def shape_selection(mode, channel, obs):
    if mode == 'cnn':
        return T.tensor([obs]).float().view(channel, 1, obs.shape[0], obs.shape[1])

    elif mode == 'non_cnn':
        return np.reshape(obs, (1, obs.size))


def size_selection(mode, obs):
    if mode == 'cnn':
        return T.tensor([obs]).float().shape

    elif mode == 'non_cnn':
        return [obs.size]


if __name__ == '__main__':
    dirname = 'result/'
    modelname = 'model_store/'
    env = gym.make('water_envs-v0')
    num_of_channel = 1
    map_size = size_selection(mode=mode, obs=env.observe())
    agent = Agent(gamma=0.9, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=map_size, lr=0.003)
    eps_hist, reward_list = [], [np.inf]
    n_games = 1500
    start_time = datetime.datetime.now()
    for i in range(n_games):
        score = 0
        done = False
        observation = shape_selection(mode=mode, channel=num_of_channel, obs=env.reset())
        while not done:
            action = agent.choose_action(observation)
            trace, state, reward, observation_, done = env.step(action)
            observation_ = shape_selection(mode=mode, channel=num_of_channel, obs=observation_)
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        reward_list.append(reward)
        if reward_list[-1] >= reward_list[-2]:
            trace_list = trace
        eps_hist.append(agent.epsilon)
        avg_score = np.mean(reward_list[-50:])
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        print('epispde', i, 'score %.2f' % reward,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon,
              'time', t)
    x = [i for i in range(n_games)]
    clip_reward = 'clip_yes'
    if clip_reward == 'clip_yes':
        clip = [-1, 0, -1, 1]
    elif clip_reward == 'clip_no':
        clip = [-1, -0.3, -0.5, 1]

    # for visualising
    comptime = comp_time(t)
    filename = dirname + 'dqn_{}_{}_{}_{}_{}.png'.format(mode, clip_reward,env.MAP_shape[0], env.MAP_shape[1], comptime)
    plotLearning(x, reward_list[1:], eps_hist, env.observe(), np.round(avg_score, 2), comptime, filename)
    filename_maze = dirname + 'dqn_maze_{}_{}_{}_{}_{}.png'.format(mode, clip_reward, env.MAP_shape[0], env.MAP_shape[1], comptime)
    show_maze(env.observe(), trace_list, filename_maze)

    # for model saving
    dt_now = datetime.datetime.now()
    file_time = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
    m = agent.Q_eval.to('cpu').state_dict()
    model_pth = modelname + 'model_{}_{}_{}_{}.pth'.format(env.MAP_shape[0],env.MAP_shape[1], mode, file_time)
    T.save(m, model_pth)

    # for saving overall scores
    f = open('score.txt', 'a', encoding='UTF-8')
    f.write('===================================================\n')
    f.write('DQN {} {} {} {} {}\n'.format(mode, clip_reward, env.MAP_shape[0], env.MAP_shape[1], comptime))
    f.write('Mode:{},Activation:Relu, MAP size:{}×{}, Computation time:{}\n'.format(mode, env.MAP_shape[0], env.MAP_shape[1], comptime))
    f.write('Step:{}, Score:{}\n'.format(n_games, avg_score))
    f.write('Water:{}, Free:{}, Wall:{}, Goal:{}\n'.format(clip[0], clip[1], clip[2], clip[3]))
    f.write('===================================================\n')
    f.close()