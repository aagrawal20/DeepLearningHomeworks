import random
import math
import operator
import numpy as np
import tensorflow as tf
import collections
import sys

Transition = collections.namedtuple('Transition', ('old_state', 'action', 'cur_state', 'cur_reward', 'next_state', 'next_reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity
        
    def encode_sample(self, idxes):
        old_state, actions, cur_state, cur_reward, next_state, next_reward = [], [], [], [], [], []
        for i in idxes:
            data = self.memory[i]
            _, old_s, action, cur_s, cur_r, next_s, next_r = data
            
            old_state.append(np.array(old_s, copy=False))
            actions.append(np.array(action, copy=False))
            cur_state.append(np.array(cur_s, copy=False))
            cur_reward.append(cur_r)
            next_state.append(np.array(next_s, copy=False))
            next_reward.append(next_r)
            
        return np.array(old_state), np.array(actions), np.array(cur_state), np.array(cur_reward), np.array(next_state), np.array(next_reward)
        

    def sample(self, batch_size):
        idexs = [random.randint(0, len(self.memory)-1) for _ in range(batch_size)]
        return self.encode_sample(idexs)

    def __len__(self):
        return len(self.memory)
    
class PrioritizedReplayBuffer(ReplayMemory):
    """ Inherits from baselines Prioritized Replay Buffer class"""
    
    def __init__(self, size, alpha):
        
        super(PrioritizedReplayBuffer, self).__init__(size)
        
        assert alpha >= 0
        self.alpha = alpha

        init_capacity = 1
        
        while init_capacity < size:
            init_capacity *= 2

        self.init_sum = SumSegmentTree(init_capacity)
        self.init_min = MinSegmentTree(init_capacity)
        
        self.max_priority = 1.0
        
    def push(self, *args):
        
        idx = self.position
        super().push(self, *args)
        self.init_sum[idx] = self.max_priority ** self.alpha
        self.init_min[idx] = self.max_priority ** self.alpha

    def proportional_sample(self, batch_size):
        res = []
        total_prob = self.init_sum.sum(0, len(self.memory) - 1)
        every_range_len = total_prob / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self.init_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self.proportional_sample(batch_size)

        weights = []
        min_prob = self.init_min.min() / self.init_sum.sum()
        max_weight = (min_prob * len(self.memory)) ** (-beta)

        for idx in idxes:
            sample_prob = self.init_sum[idx] / self.init_sum.sum()
            weight = (sample_prob * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self.encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)
            self.init_sum[idx] = priority ** self.alpha
            self.init_min[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
    
    

def select_eps_greedy_action(session, input, policy_model, obs, step, num_actions, EPS_START, EPS_END, EPS_DECAY, exploit, explore):
    """
    Decides whether the agent should exploit or explore

    :param policy_model: (callable): mapping of `obs` to q-values
    :param obs: (np.array): current state observation
    :param step: (int): training step count
    :param num_actions: (int): number of actions available to the agent
    :param EPS_START:
    :param EPS_END:
    :param EPS_DECAY:
    :return: action
    """

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)
    # exploit
    if random.random() > eps_threshold:
        exploit += 1
        output = session.run([policy_model.output], feed_dict={input:obs})
        action = np.argmax(output[0])
    # explore
    else:
        explore += 1
        action = random.randrange(num_actions)
      

    return action, explore, exploit

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
