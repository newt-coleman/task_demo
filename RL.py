import numpy as np
from scipy.optimize import minimize

class FeatureRLDecay():
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha  # learning rate
        self.beta = beta    # softmax temp
        self.gamma = gamma  # decay rate

        self.actions = [] ## placeholder for set_action()
        self.v_feature = {(k//4, k%4) : 0 for k in range(12) }
        self.q = np.zeros(4)

    def reset_v(self):
        self.v_feature = {(k//4, k%4) : 0 for k in range(12) }
    
    def set_action(self, action_set):
        self.actions = action_set  ## actions available at a time step

    def boltz_prob(self):
        """generates the boltzman probability for each action option"""
        temperature = max(self.beta, 0.01) 
        p_a = np.exp(temperature*self.q) / np.sum(np.exp(temperature*self.q))
        return p_a 
    
    def set_q(self):
        self.q = np.zeros(4)
        for idx in range(len(self.q)):
            action = self.actions[idx]

            self.q[idx] += self.v_feature[(0, action[0])]
            self.q[idx] += self.v_feature[(1, action[1])]
            self.q[idx] += self.v_feature[(2, action[2])]
    
    def select_stim(self):
        ## define q func
        self.set_q()

        ## calculate probability
        p_a = self.boltz_prob()

        a_k_idx = np.random.choice(list(range(4)), p=p_a)
        return a_k_idx

    def update_v(self, a_k_idx, r):
        a_k = self.actions[a_k_idx] # FIX THIS SHAPE ISSUE BW TRAIN AND PREDICT
        if a_k.shape == (1,1):      # this is the fix??
            a_k = a_k[0][0]
        self.set_q()

        for z in self.v_feature.keys():
            has_feature = (z[0] == 0 and z[1] == a_k[0]) or (z[0] == 1 and z[1] == a_k[1]) or (z[0] == 2 and z[1] == a_k[2])
            if has_feature and r==1:
                self.v_feature[z] += self.alpha*((r)-self.q[a_k_idx])
            else:
                # print(self.v_feature[z])
                self.v_feature[z] *= (1-self.gamma)
                # print(self.v_feature[z])
