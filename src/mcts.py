import math
import pickle
import numpy as np
from env import Env
from dialogue import DialogSession


class MCTS():
    def __init__(self, env: Env, num_models, Q_0, cuct, gamma, max_depth=8, enc=None, lambda1=0):
        self.env = env

        self.Ns: dict = {}
        self.Nsa: dict = {}
        self.Q: dict = {}

        self.num_models = num_models
        self.Q_0 = Q_0
        self.cuct = cuct
        self.gamma = gamma
        self.max_depth = max_depth

        self.enc = enc

        # Token prices (USD per 1 million tokens), ordered by candidate LLM: [LLM1, LLM2, ...].
        self.input_cost = [0.459, 0.28, 1.25]
        self.input_cache = [0.0918, 0.028, 0.125]
        self.output_cost = [1.836, 0.42, 10]
        self.block_size = 128

        # To use cost reward set self.lambda1 = -0.00005
        self.lambda1 = lambda1

    
    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text, disallowed_special=()))
    
    def compute_cost(self, state: DialogSession):
        hashable_state = self._to_string_rep(state)
        if hashable_state == '':
            return 0
        ans_str = state.history[-2][2]
        query_str = state.history[-3][2]
        ans_l = self.count_tokens(ans_str)
        query_l = self.count_tokens(query_str)
        if len(hashable_state) == 1:
            cost = self.input_cost[int(hashable_state)] * query_l + self.output_cost[int(hashable_state)] * ans_l
            return self.lambda1 * cost
        else:
            m_t = int(hashable_state[-1])
            m_p = int(hashable_state[-4])
            h_l = 0
            for h in state.history[:-3]:
                h_l += self.count_tokens(h[2])
            if m_t != m_p:
                cost = self.input_cost[m_t] * (query_l + h_l) + self.output_cost[m_t] * ans_l
            else:
                cost = self.input_cost[m_t] * (h_l % self.block_size + query_l) + self.input_cache[m_t] * ((h_l // self.block_size) * self.block_size) + self.output_cost[m_t] * ans_l
            return self.lambda1 * cost

    def _to_string_rep(self, state: DialogSession):
        das = []
        for (speaker, da, _) in state:
            if speaker == state.SYS:
                das.append(str(da))
        return "__".join(das)

    def _init_node(self, state: DialogSession, prev_C, C_t=None, done=None):
        hashable_state = self._to_string_rep(state)

        self.Ns[hashable_state] = 0
        self.Nsa[hashable_state] = {action: 0 for action in range(self.num_models)}
        self.Q[hashable_state] = {action: self.Q_0 for action in range(self.num_models)}
        if hashable_state not in self.env.realizations:
            self.env.realizations[hashable_state] = (state.copy(), C_t, done)

        v = self.rollout(state.copy(), prev_C, C_t, done, 0)
        return v
    
    def rollout(self, state: DialogSession, prev_C, C_t=None, done=None, depth=None):
        if C_t == None:
            C_t, done = self.env.compute_reward_sim(state)
        self._add_new_realizations(state, C_t, done)
        cost_t = self.compute_cost(state)
        r_t = C_t - prev_C + cost_t
        if done:
            return r_t
        if self.gamma ** depth < 1e-4 or depth >= self.max_depth:
            return r_t
        
        action = self.greedy_rollout(state.copy())

        next_state, next_C_t, next_done = self._get_next_state(state, action)
        return r_t + self.gamma * self.rollout(next_state, C_t, next_C_t, next_done, depth + 1)
    
    def _add_new_realizations(self, state: DialogSession, C_t=None, done=None):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.env.realizations:
            self.env.realizations[hashable_state] = (state.copy(), C_t, done)
        
    def _get_next_state(self, state: DialogSession, best_action):
        if self._to_string_rep(state) == '':
            prefetch_state = str(best_action)
        else:
            prefetch_state = self._to_string_rep(state) + "__" + str(best_action)
        if prefetch_state in self.env.realizations:
            return self.env.realizations[prefetch_state]

        next_state = self.env.step_sim(state.copy(), best_action)
        return next_state, None, None

    def search(self, state: DialogSession, prev_C, C_t=None, done=None):
        hashable_state = self._to_string_rep(state)

        if C_t == None:
            C_t, done = self.env.compute_reward_sim(state)
        r_t = C_t - prev_C + self.compute_cost(state)

        if done:
            return r_t

        if hashable_state not in self.Ns:
            v = self._init_node(state, prev_C, C_t, done)
            return v
        else:
            self._add_new_realizations(state, C_t, done)

        best_uct, best_action = -float('inf'), -1
        for a in range(self.num_models):
            Ns = self.Ns[hashable_state]
            if Ns == 0:
                Ns = 1e-8
            if self.Nsa[hashable_state][a] == 0:
                uct = float('inf')
            else:
                uct = self.Q[hashable_state][a] + self.cuct * math.sqrt(math.log(max(Ns, 1)) / (1 + self.Nsa[hashable_state][a]))
            if uct > best_uct or (uct == best_uct and np.random.rand() < 0.5):
                best_uct, best_action = uct, a

        next_state, next_C_t, next_done = self._get_next_state(state, best_action)

        v = self.search(next_state, C_t, next_C_t, next_done)

        G_t = r_t + self.gamma * v

        self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + G_t) / (self.Nsa[hashable_state][best_action] + 1)
        self.Ns[hashable_state] += 1
        self.Nsa[hashable_state][best_action] += 1

        return G_t
    
    def get_action_prob(self, state: DialogSession, prev_C, C_t=None, done=None):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.Ns:
            self._init_node(state, prev_C, C_t, done)
        prob = np.array([self.Q[hashable_state][a] for a in range(self.num_models)])
        return prob
    
    def save_mcts(self, filename):
        mcts = {'Q': self.Q, 'Ns': self.Ns, 'Nsa': self.Nsa, 'realizations': self.env.realizations}
        with open(filename, 'wb') as file:
            pickle.dump(mcts, file)
    
    def greedy_rollout(self, state: DialogSession):
        scores = []
        for a in range(self.num_models):
            next_state, next_C_t, next_done = self._get_next_state(state, a)
            if next_C_t == None:
                next_C_t, next_done = self.env.compute_reward_sim(next_state)
            self._add_new_realizations(next_state, next_C_t, next_done)
            cost_t = self.compute_cost(next_state)
            scores.append(next_C_t + cost_t)
        scores = np.array(scores, dtype=float)
        max_indices = np.flatnonzero(scores == scores.max())
        action = np.random.choice(max_indices)
        return action
    