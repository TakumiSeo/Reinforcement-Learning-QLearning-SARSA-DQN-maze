# Reinforcement-Learning-in-the-original-environment-maze-
/
README.md

for Q-Learing, SARSA, DQN

## Requirement 
- python3.6
- OpenAI Gym

## install_requires
['gym'
 'numpy', 
 'matplotlib', 
 'scikit-image',
 'pytorch',
 'keras',
 'abstractmethod']
 
 ## Environment
This is for test.
The agent is supposed to avoid entering water whose colour is skyblue and then make a walk to the goal located the most right below.
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gym
import water_envs
import numpy as np

env = gym.make('water_envs-v0')

# just for random choice
import random
STEP = 10
actions = [2, 2, 2, 2]
for i in range(STEP):
    trace, state, reward, obs, done = env.step(random.choice(actions))
    print(point)
print('obs:{0}, reward:{1}'.format(obs, reward))
```
## Results
![5_5_sa_map](https://user-images.githubusercontent.com/49015441/101862361-8582e700-3bb5-11eb-8a48-0f3ba9257021.png)
![5_5_ql](https://user-images.githubusercontent.com/49015441/101862365-874caa80-3bb5-11eb-82e8-7752c70a34ff.png)
![30_30_ql](https://user-images.githubusercontent.com/49015441/101862382-92073f80-3bb5-11eb-878b-deaafd774bf6.png)
![30_30_ql_reward](https://user-images.githubusercontent.com/49015441/101862388-94699980-3bb5-11eb-9f37-170097915be2.png)
