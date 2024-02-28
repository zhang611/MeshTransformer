import numpy as np
import matplotlib.pyplot as plot


def getPositionEncoding(seq_len, dim, n=10000):
    PE = np.zeros(shape=(seq_len, dim))
    for pos in range(seq_len):
        for i in range(int(dim / 2)):
            denominator = np.power(n, 2 * i / dim)
            PE[pos, 2 * i] = np.sin(pos / denominator)
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)

    return PE


PE = getPositionEncoding(seq_len=300, dim=256, n=10000)
print(PE.shape)
print(PE)


