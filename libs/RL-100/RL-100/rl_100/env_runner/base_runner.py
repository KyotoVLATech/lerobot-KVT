from typing import Dict
from rl_100.policy.base_policy import BasePolicy


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

    def make_env(self, record_video=True):
        """Create and return a new independent environment instance."""
        raise NotImplementedError()
