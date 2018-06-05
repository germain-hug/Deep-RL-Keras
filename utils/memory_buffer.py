import random
import numpy as np

from collections import deque

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue
    """
    def __init__(self, buffer_size):
        self.count = 0
        self.buffer = deque()
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state):
        experience = (state, action, reward, done, new_state)
        # Check if buffer is already full
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])

        return s_batch, a_batch, r_batch, d_batch, new_s_batch

    def clear(self):
        self.buffer = deque()
        self.count = 0
