import numpy as np
import torch
from memory_profiler import profile


@profile
def main():
    print('hello')
    arr1 = np.random.random((5000, 128, 256))

    arr2 = torch.tensor(arr1)
    arr2 = arr2.to('cuda')
    print('test')

if __name__ == '__main__':
    main()
