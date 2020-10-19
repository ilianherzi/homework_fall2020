'''GOAL
# Setup policy and rollout workers
env = gym.make("CartPole-v0")
policy = CustomPolicy(env.observation_space, env.action_space, {})
workers = WorkerSet(
    policy_class=CustomPolicy,
    env_creator=lambda c: gym.make("CartPole-v0"),
    num_workers=10)

while True:
    # Gather a batch of samples
    T1 = SampleBatch.concat_samples(
        ray.get([w.sample.remote() for w in workers.remote_workers()]))

    # Improve the policy using the T1 batch
    policy.learn_on_batch(T1)

    # Broadcast weights to the policy evaluation workers
    weights = ray.put({"default_policy": policy.get_weights()})
    for w in workers.remote_workers():
        w.set_weights.remote(weights)

'''

import gym
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel

config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 4,
          'train_batch_size': 1000,
          'model': {
              'fcnet_hiddens': [128, 128]
          }}

trainer = DQNTrainer(env="LunarLander-v3", config=config).with_updates(
    execution_plan=execution_plan
)
results=trainer.train() # once enough data is collected the model is updated and the results are returned

from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.agents.dqn.simple_q_tf_policy import SimpleQTFPolicy
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.policy.policy import LEARNER_STATS_KEY, Policy
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator
def execution_plan(workers: WorkerSet,
                   config: TrainerConfigDict) -> LocalIterator[dict]:
    if config.get("prioritized_replay"):
        prio_args = {
            "prioritized_replay_alpha": config["prioritized_replay_alpha"],
            "prioritized_replay_beta": config["prioritized_replay_beta"],
            "prioritized_replay_eps": config["prioritized_replay_eps"],
        }
    else:
        prio_args = {}

    local_replay_buffer = LocalReplayBuffer(
        num_shards=1,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        replay_batch_size=config["train_batch_size"],
        replay_mode=config["multiagent"]["replay_mode"],
        replay_sequence_length=config["replay_sequence_length"],
        **prio_args)

    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our local replay buffer. Calling
    # next() on store_op drives this.
    store_op = rollouts.for_each(
        StoreToReplayBuffer(local_buffer=local_replay_buffer))

    def update_prio(item):
        samples, info_dict = item
        if config.get("prioritized_replay"):
            prio_dict = {}
            for policy_id, info in info_dict.items():
                # TODO(sven): This is currently structured differently for
                #  torch/tf. Clean up these results/info dicts across
                #  policies (note: fixing this in torch_policy.py will
                #  break e.g. DDPPO!).
                td_error = info.get("td_error",
                                    info[LEARNER_STATS_KEY].get("td_error"))
                prio_dict[policy_id] = (samples.policy_batches[policy_id]
                                        .data.get("batch_indexes"), td_error)
            local_replay_buffer.update_priorities(prio_dict)
        return info_dict

    # (2) Read and train on experiences from the replay buffer. Every batch
    # returned from the LocalReplay() iterator is passed to TrainOneStep to
    # take a SGD step, and then we decide whether to update the target network.
    post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
    replay_op = Replay(local_buffer=local_replay_buffer) \
        .for_each(lambda x: post_fn(x, workers, config)) \
        .for_each(TrainOneStep(workers)) \
        .for_each(update_prio) \
        .for_each(UpdateTargetNetwork(
            workers, config["target_network_update_freq"]))

    # Alternate deterministically between (1) and (2). Only return the output
    # of (2) since training metrics are not available until (2) runs.
    train_op = Concurrently(
        [store_op, replay_op],
        mode="round_robin",
        output_indexes=[1],
        round_robin_weights=calculate_rr_weights(config))
    lst = list(StandardMetricsReporting(train_op, workers, config))
    print(lst[:10])
    raise Exception
    return StandardMetricsReporting(train_op, workers, config) #default config "timesteps_per_iteration"=1000, min_iter_time_s": 1,

def calculate_rr_weights(config: TrainerConfigDict):
    if not config["training_intensity"]:
        return [1, 1]
    # e.g., 32 / 4 -> native ratio of 8.0
    native_ratio = (
        config["train_batch_size"] / config["rollout_fragment_length"])
    # Training intensity is specified in terms of
    # (steps_replayed / steps_sampled), so adjust for the native ratio.
    weights = [1, config["training_intensity"] / native_ratio]
    return weights
