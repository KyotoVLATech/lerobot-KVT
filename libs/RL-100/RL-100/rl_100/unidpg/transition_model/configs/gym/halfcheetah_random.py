from copy import deepcopy
from rl_100.unidpg.transition_model.configs.gym.default import default_args


halfcheetah_random_args = deepcopy(default_args)
halfcheetah_random_args["rollout_length"] = 5
halfcheetah_random_args["penalty_coef"] = 0.5