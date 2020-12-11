# Own-Maze-OpenAI-gym-
for Q-Learing, SARSA, DQN

## Requirement 
- python3.7
- OpenAI Gym

##install_requires
['gym'
 'numpy', 
 'matplotlib', 
 'scikit-image',
 'pytorch',
 'keras',
 'abstractmethod']
 
 ## Environment
for test
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
    point, reward, obs, done = env.step(random.choice(actions))
    print(point)
print('obs:{0}, reward:{1}'.format(obs, reward))
```
