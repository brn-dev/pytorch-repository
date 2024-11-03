# import numpy as np
# import torch
# from memory_profiler import profile
# import cProfile as cp
#
# arr1 = np.random.random((5000, 128, 256))
#
# @profile
# def main():
#     # print('hello')
#     shape = arr1.shape
#     arr2 = arr1.reshape((shape[0] * shape[1], *shape[2:]))
#
#     arr3 = arr2[:100]
#     arr4 = arr2[100:200]
#
#     x = arr3[np.array([0, 1, 2, 3, 4, 5])]
#     y = arr4[np.array([0, 1, 2, 3, 4, 5])]
#
#     # print('test')
#
# if __name__ == '__main__':
#     prof = cp.Profile()
#     prof.enable()
#     for _ in range(1000):
#         main()
#     prof.disable()
#     prof.print_stats()
#     prof.dump_stats('tmp_profile')
from src.reinforcement_learning.core.buffers.replay.ring_with_reservoir_replay_buffer import RingWithReservoirReplayBuffer

if __name__ == '__main__':
    RingWithReservoirReplayBuffer.for_env()
