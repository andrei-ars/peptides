import os
from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
from enviroment import Enviroment
import random

# configurations
#env = gym.make("CartPole-v0")
max_episodes = 5000 # 100000
num_episodes_before_update = 50 # 50 #100
max_steps = 66666 #100
solved_reward = 190
solved_repeat = 5

model_dir = "saved_model"
network_map = {"restorable_model_1": "file_name_1", "qnet_target": "model1.pt"}
os.system("mkdir -p {}".format(model_dir))


class QNet(nn.Module):
    """ Model definition
    """
    def __init__(self, state_dim, action_num):
        super(QNet, self).__init__()

        n_hidden = 32
        self.fc1 = nn.Linear(state_dim, n_hidden)
        #self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        #a = t.relu(self.fc2(a))
        return self.fc3(a)


def start_learning(continue_training=False):

    env = Enviroment()
    observe_dim = env.obs_size() #3*15 # 30
    action_num = env.number_of_actions()

    q_net = QNet(observe_dim, action_num).float()
    q_net_t = QNet(observe_dim, action_num).float()
    dqn = DQN(q_net, q_net_t,
              t.optim.Adam,
              nn.MSELoss(reduction='sum'),
              learning_rate=0.003,  # 0.001
              replay_device='cpu')

    if continue_training:
        print("Load the model")
        dqn.load(model_dir, network_map=network_map)
 
    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise(
                    {"state": old_state}
                )
                #action = t.tensor([[ random.randint(0, action_num-1) ]])
                
                state, reward, terminal, _ = env.step(action.item())

                #print("{}: {} - reward={}".format(int(action), state[:15], reward))
                #if terminal:
                #    print("step={}".format(step))
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                dqn.store_transition({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps
                })
        print(terminal, step)

        # update, update more if episode is longer, else less
        if episode > num_episodes_before_update or continue_training:
            for _ in range(step):
                dqn.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 + total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))
        #env.driver.logger.log("Episode {} total reward={:.2f}"
        #            .format(episode, smoothed_total_reward))

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0

    #dqn.save(model_dir, network_map=network_map)
    #print("Saved with network_map={}".format(network_map))

if __name__ == "__main__":

    start_learning()