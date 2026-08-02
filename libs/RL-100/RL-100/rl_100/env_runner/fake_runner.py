"""実機/シムを持たないタスク(sushi等)用の軽量ダミー runner.

pusht_runner 等は mujoco / pytorch3d / gym をトップレベルで import するため、
オフライン学習だけしたい実機データセットでは import 自体が重い依存を要求してしまう。
本 runner は何も重いものを import せず、`run()` はダミーの評価指標を返すだけ。
`env` は None（オフライン学習ではオンライン評価を行わない前提）。

実評価・オンラインRLは iloha_online_rl.py で別途行う。
"""

import numpy as np

from rl_100.env_runner.base_runner import BaseRunner


class FakeRunner(BaseRunner):
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(output_dir)
        self.env = None
        # 他のコードが参照しうる属性だけ保持（存在しないと AttributeError になる箇所向け）
        self.eval_episodes = kwargs.get("eval_episodes", 1)
        self.max_steps = kwargs.get("max_steps", 1)
        self.n_obs_steps = kwargs.get("n_obs_steps", 1)
        self.n_action_steps = kwargs.get("n_action_steps", 1)
        self.env_num = kwargs.get("env_num", 1)
        self.fps = kwargs.get("fps", 30)

    def _dummy_log(self):
        return {
            "test_mean_score": 0.0,
            "mean_returns": 0.0,
            "mean_success_rates": 0.0,
            "mean_n_goal_achieved": 0.0,
            "SR_test_L3": 0.0,
            "SR_test_L5": 0.0,
        }

    def run(self, policy, *args, **kwargs):
        return self._dummy_log()

    def run1(self, policy, *args, **kwargs):
        return self._dummy_log()

    def make_env(self, record_video=True):
        return None
