import numpy as np
import os
import errno
from scipy.optimize import minimize
import RL
from scipy.stats import f_oneway, describe

class Subject():
    
    def _load_split(self):
        """loads data into 3 levels of probability, then splits into a testing and training dataset"""
        if os.path.exists(os.path.join(os.getcwd(), self.data_path)) is False:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.data_path)
        
        envs = [[] for k in range(len(self.p_levels))] 

        rd = [[] for k in range(len(self.p_levels))] 

        # load in data by probability condition
        for file_name in os.listdir(self.data_path):
            dat = np.load(self.data_path + "\\" + file_name, allow_pickle=True)
            p = float(file_name.split("_")[1])
            idx = np.argwhere(self.p_levels==p)[0]

            if 'choicedata' in file_name:
                envs[idx].append(dat)
            else:
                rd[idx].append(dat)

        
        # split data into train and test sets
        # test_args = np.random.choice(list(range(0, len(rd[0]))), 
        #                              size = int(0.3*len(rd[0])), replace =False)
        # for i in range(len(rd[0])):
        #     if i in test_args:
        #         for j in range(len(self.p_levels)):
        #             self.test_envs[j].append(envs[j][i])
        #             self.test_rd[j].append(rd[j][i])
        #     else:
        #         for j in range(len(self.p_levels)):
        #             self.train_envs[j].append(envs[j][i])
        #             self.train_rd[j].append(rd[j][i])
        # print(len(self.train_envs[0]))

        ## all train all test
        for i in range(len(rd[0])):
            for j in range(len(self.p_levels)):
                self.test_envs[j].append(envs[j][i])
                self.test_rd[j].append(rd[j][i])
        self.train_envs = self.test_envs
        self.train_rd = self.test_rd
 
        # print(len(self.train_envs[0]))


    def __init__(self, data_path, p_levels):
        self.data_path = data_path ## name of directory with data
        self.p_levels = p_levels

        self.train_envs = [[] for k in range(len(self.p_levels))] # lol
        self.train_rd = [[] for k in range(len(self.p_levels))]

        self.test_envs = [[] for k in range(len(self.p_levels))]
        self.test_rd = [[] for k in range(len(self.p_levels))]

        self._load_split()

        self.alpha = 0
        self.beta = 0
        self.gamma = 0

    def _set_params(self, a, b, g):
        self.alpha = a
        self.beta = b
        self.gamma = g

    def _argchoice(self, v, e):
        eqs = []
        for element in v:
            eqs.append(np.all(element==e))
        return np.argwhere(eqs)

    def LL(self, data_env, data_reward, alpha, beta, gamma):
        LL = 0.0
        # for each trial
        
        for trial_num, envs in enumerate(data_env):
            agent = RL.FeatureRLDecay(alpha, beta, gamma)
            for t in range(len(envs[:,0])):
                agent.set_action(envs[t, :4].reshape(-1))
                probs = agent.boltz_prob()

                real_ak_idx = self._argchoice(agent.actions, envs[t, -1])
                p = probs[real_ak_idx]

                LL += np.log(max(p, 10e-10))

                agent.update_v(real_ak_idx, data_reward[trial_num][t])

        return LL[0][0]
    
    def train(self):
        env = []
        reward_data = []
        for i in range(len(self.p_levels)):
            for j in range(len(self.train_rd[0])):
                env.append(self.train_envs[i][j])
                reward_data.append(self.train_rd[i][j])
        # print(env[0].shape)

        def loss(theta):
            """Loss function = 1 * log-likelihood"""
            alpha, beta, gamma = theta
            # intermediate.append([alpha, beta, gamma])

            return -1 * self.LL(env, reward_data, alpha, beta, gamma)
        
        print("Start!")
        res = minimize(loss, x0 = [0.8, 15, 1], 
                 bounds = [(0, 1), (0, None), (0,1)], method='Nelder-Mead')
        print(res.success)

        self.alpha, self.beta, self.gamma = res.x 
        return self.alpha, self.beta, self.gamma
    
    def predict_RL(self):
         # 1 for accurate prediction, 0 for not
        rl_choices = [[] for k in range(len(self.p_levels))]
        agent = RL.FeatureRLDecay(self.alpha, self.beta, self.gamma)

        for p in range(len(self.p_levels)):
            for trial in range(len(self.test_envs[p])):
                agent.reset_v()

                env = self.test_envs[p][trial]
                rd = self.test_rd[p][trial]

                for k in range(len(rd)):
                    agent.set_action(env[k, :4])
                    a_k_idx = agent.select_stim()
                    real_ak_idx = self._argchoice(agent.actions, env[k, -1])[0][0]

                    if a_k_idx == real_ak_idx:
                        rl_choices[p].append(1)
                    else:
                        rl_choices[p].append(0)
                    agent.update_v(real_ak_idx, rd[k])
        return rl_choices
    
    def predict_SA(self):
        sa_choices = [[] for k in range(len(self.p_levels))]
        stimuli = np.zeros(64, dtype=tuple)

        for col in range(4):
            for shape in range(4):
                for pat in range(4):
                    stimuli[int(16*col + 4*shape + pat)] = (col, shape, pat)

        for p in range(len(self.p_levels)):
            for trial in range(len(self.test_envs[p])):
                sa_dict = {stimuli[i] : 0 for i in range(64)}

                env = self.test_envs[p][trial]
                rd = self.test_rd[p][trial]

                for t in range(len(rd)):
                    # extract associ of options
                    vals = np.array([sa_dict[tuple(env[t, 0])], sa_dict[tuple(env[t, 1])], 
                                         sa_dict[tuple(env[t, 2])], sa_dict[tuple(env[t, 3])]])
                    if np.sum(vals) == 0:
                        p_a = np.ones(4)*0.25
                    else:
                        p_a = vals / np.sum(vals)
                    a_k_idx = np.random.choice(list(range(4)), p=p_a)
                    real_ak_idx = self._argchoice(env[t, :4], env[t, -1])[0][0]

                    if a_k_idx == real_ak_idx:
                        sa_choices[p].append(1)
                    else:
                        sa_choices[p].append(0)

                    if rd[t] == 1:
                        sa_dict[tuple(env[t, real_ak_idx])] += 50

        return sa_choices

# you = Subject("bj", np.array([0.6, 0.8, 1.]))
# you.train()
# print(you.alpha)
# print(you.beta)
# print(you.gamma)

# acc = you.predict_RL()
# print(sum(acc[0]) / len(acc[0]))
# print(sum(acc[1]) / len(acc[1]))
# print(sum(acc[2]) / len(acc[2]))

# sacc = you.predict_SA()
# print(sum(sacc[0]) / len(sacc[0]))
# print(sum(sacc[1]) / len(sacc[1]))
# print(sum(sacc[2]) / len(sacc[2]))


