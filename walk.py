import numpy as np
import os


# ------------------------------------------------------ #
# ---------------------得到随机游走序列-------------------- #
# ------------------------begin------------------------- #

def get_seq_random_walk_random_global_jumps(mesh_extra, f0, seq_len):
    nbrs = mesh_extra['edges']
    n_vertices = mesh_extra['n_vertices']
    seq = np.zeros((seq_len + 1,), dtype=np.int32)
    jumps = np.zeros((seq_len + 1,), dtype=np.bool)
    visited = np.zeros((n_vertices + 1,), dtype=np.bool)
    visited[-1] = True
    visited[f0] = True
    seq[0] = f0
    jumps[0] = [True]
    backward_steps = 1
    jump_prob = 1 / 100
    for i in range(1, seq_len + 1):
        this_nbrs = nbrs[seq[i - 1]]  # 当前点的邻边
        nodes_to_consider = [n for n in this_nbrs if not visited[n]]  # 可选邻边
        jump_now = np.random.binomial(1, jump_prob)
        # 有可选边且不jump，边加入序列，回退计数+1
        if len(nodes_to_consider) and not jump_now:
            to_add = np.random.choice(nodes_to_consider)  # 随机选取下个点
            jump = False
            backward_steps = 1
        else:
            if i > backward_steps and not jump_now:
                to_add = seq[i - backward_steps - 1]
                backward_steps += 2
            else:  # jump！！！一个序列中并不一定是连续的边序列，若有jump则不连续
                backward_steps = 1
                to_add = np.random.randint(n_vertices)  # 随机选取新的起点
                jump = True
                visited[...] = 0  # 清空访问
                visited[-1] = True
        visited[to_add] = 1
        seq[i] = to_add
        jumps[i] = jump

    return seq, jumps

# ------------------------------------------------------ #
# ---------------------得到随机游走序列-------------------- #
# --------------------------end------------------------- #
