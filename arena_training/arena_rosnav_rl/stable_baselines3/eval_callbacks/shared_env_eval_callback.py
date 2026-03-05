"""
SharedEnvEvalCallback – runs periodic evaluation on the *same* VecEnv that
is used for training, eliminating the need for a duplicate set of subprocess
workers.

Before each evaluation run the callback switches the underlying environments
to "eval mode" by pushing ``eval_max_steps`` to every worker via
``VecEnv.set_attr``.  After evaluation finishes it restores
``train_max_steps`` so normal rollout collection is unaffected.
"""

from rosnav_rl.utils.stable_baselines3.callbacks import RosnavEvalCallback


class SharedEnvEvalCallback(RosnavEvalCallback):
    """Eval callback that reuses the training VecEnv for evaluation.

    Parameters
    ----------
    train_max_steps:
        ``_max_steps_per_episode`` value used during normal training rollouts.
    eval_max_steps:
        ``_max_steps_per_episode`` value to use during evaluation episodes
        (typically larger to allow the agent to complete longer tasks).
    **kwargs:
        All other arguments are forwarded to :class:`RosnavEvalCallback`.
    """

    def __init__(self, train_max_steps: int, eval_max_steps: int, **kwargs):
        super().__init__(**kwargs)
        self._train_max_steps = train_max_steps
        self._eval_max_steps = eval_max_steps

    def _on_step(self) -> bool:
        # Mirror the condition used inside EvalCallback._on_step so we only
        # touch the envs when an evaluation is actually about to happen.
        will_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0

        if will_eval:
            self.eval_env.set_attr("_max_steps_per_episode", self._eval_max_steps)

        result = super()._on_step()

        if will_eval:
            self.eval_env.set_attr("_max_steps_per_episode", self._train_max_steps)

        return result
