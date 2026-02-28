import numpy as np

class FeatureRLDecay():
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha  # learning rate
        self.beta = beta    # softmax temp
        self.gamma = gamma  # decay rate

        self.actions = [] ## placeholder for set_action()
        self.v_feature = {(k//4, k%4) : 0 for k in range(12) }
        self.q = np.empty(4, dtype=tuple)

    def reset_v(self):
        self.v_feature = {(k//4, k%4) : 0 for k in range(12) }
    
    def set_action(self, action_set):
        self.actions = action_set  ## actions available at a time step

    def boltz_prob(self):
        """generates the boltzman probability for each action option"""
        p_all = 0
        for j in range(self.actions):
            p_all += np.eps(self.beta*self.q[j])
        p_a = np.empty(4)
        for j in range(self.actions):
            p_a[j] = np.eps(self.beta*self.q[j]) / p_all
        return p_a 
    
    def select_stim(self):
        ## define q func
        self.q = np.empty(4, dtype=tuple)
        for idx in range(len(self.q)):
            for z in range(len(self.actions[idx])):
                self.q[idx] += self.v_feature[z]

        ## calculate probability
        p_a = self.boltz_prob()

        a_k_idx = np.random.choice(list(range(4)), p_a)
        return a_k_idx

    def update_v(self, a_k_idx, r):
        a_k = self.actions[a_k_idx]
        for z in self.v_feature.keys():
            if np.all(z==a_k[0]) or np.all(z==a_k[1]) or np.all(z==a_k[2]):
                self.v_feature[z] += self.alpha*(r-self.q[a_k_idx])
            else:
                self.v_feature[z] *= self.gamma


def LL(data_env, data_reward, alpha, beta, gamma):
    LL = 0.0
    # for each trial
    
    for trial_num, envs in enumerate(data_env):
        agent = FeatureRLDecay(alpha, beta, gamma)
        for t in range(len(envs[:,0])):
            agent.set_action(envs[t, :4])
            probs = agent.boltz_prob()
            p = probs argchoice

            LL += np.log(max(p, 10e-10))

            real_ak_idx = arg index (envs[t, -1] == envs[t, :])
            agent.update_v(real_ak_idx, data_reward[trial_num, t])

    return LL
            



      # reward table resets # assume no perseverations
      # for each choice
        # calculate and sum ll
        # update reward table
       

