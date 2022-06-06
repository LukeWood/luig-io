from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import tensorflow as tf
class config_mario:
    def __init__(self, seed):
        self.env_name = "SuperMarioBrosRandomStagesEnv-v4"
        self.record = True
        seed_str = "seed=" + str(seed)
        # output config
        self.output_path = "results/{}-{}/".format(
            self.env_name, seed_str
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 3
        self.summary_freq = 1

        self.action_dim = len(SIMPLE_MOVEMENT)

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 20000 if tf.test.is_gpu_available() else 200  # number of steps used to compute each policy update
        self.max_ep_len = 20000 if tf.test.is_gpu_available() else 200   # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 0.9999  # the discount factor
        self.normalize_advantage = True

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size



def get_config(seed=15):
    return config_mario(seed)
