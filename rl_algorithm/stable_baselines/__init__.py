from rl_algorithm.stable_baselines.a2c import A2C
from rl_algorithm.stable_baselines.acer import ACER
from rl_algorithm.stable_baselines.acktr import ACKTR
from rl_algorithm.stable_baselines.deepq import DQN
from rl_algorithm.stable_baselines.her import HER
from rl_algorithm.stable_baselines.ppo2 import PPO2
from rl_algorithm.stable_baselines.td3 import TD3
from rl_algorithm.stable_baselines.sac import SAC

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from rl_algorithm.stable_baselines.ddpg import DDPG
    from rl_algorithm.stable_baselines.gail import GAIL
    from rl_algorithm.stable_baselines.ppo1 import PPO1
    from rl_algorithm.stable_baselines.trpo_mpi import TRPO
del mpi4py

__version__ = "2.10.0a0"

