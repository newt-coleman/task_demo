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
        p_all = 0
        for j in range(len(self.actions)):
            p_all += np.exp(self.beta*self.q[j])
        p_a = np.empty(4)
        for j in range(len(self.actions)):
            p_a[j] = np.exp(self.beta*self.q[j]) / p_all
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
        a_k = np.array(self.actions[a_k_idx])[0][0]  # FIX THIS SHAPE ISSUE BW TRAIN AND PREDICT
        self.set_q()
        for z in self.v_feature.keys():
            if np.all(z==a_k[0]) or np.all(z==a_k[1]) or np.all(z==a_k[2]):
                # print("qi=" + str(self.q[a_k_idx]))
                self.v_feature[z] += self.alpha*(r-self.q[a_k_idx])
            else:
                self.v_feature[z] *= self.gamma
