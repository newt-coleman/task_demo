import numpy as np

class FeatureRLDecay():
    def __init__(self, alpha, beta, gamma, actions):
        self.alpha = alpha  # learning rate
        self.beta = beta    # softmax temp
        self.gamma = gamma  # decay rate

        self.v_feature = {(k//4, k%4) : 0 for k in range(12) }
        self.q = np.empty(4, dtype=tuple)

    def select_stim(self, actions):
        ## define q func
        for idx in range(len(self.q)):
            for z in range(len(actions[idx])):
                self.q[idx] += self.v_feature[z]

        ## calculate probability
        p_all = 0
        for j in range(actions):
            p_all += np.eps(self.beta*self.q[j])
        p_a = np.empty(4)
        for j in range(actions):
            p_a[j] = np.eps(self.beta*self.q[j]) / p_all
        a_k_idx = np.random.choice(list(range(4)))
        return a_k_idx, actions[a_k_idx]

    def update_v(self, a_k_idx, a_k, r):

        for z in self.v_feature.keys():
            if np.all(z==a_k[0]) or np.all(z==a_k[1]) or np.all(z==a_k[2]):
                self.v_feature[z] += self.alpha*(r-self.q[a_k_idx])
            else:
                self.v_feature[z] *= self.gamma

