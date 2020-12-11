from gym.envs.registration import register

register(
  id='water_envs-v0',
  entry_point='water_envs.water_no_t:Water',
)