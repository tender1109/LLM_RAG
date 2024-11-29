import numpy as np

# 成功概率和不成功概率
P = [0.1, 0.15, 0.05, 0.15]
Q = [0.1, 0.05, 0.15, 0.05, 0.2]
n = len(P)

# 初始化成本和根表格
cost = np.zeros((n + 1, n + 1))
root = np.zeros((n + 1, n + 1), dtype=int)

# 计算成本
for length in range(1, n + 1):  # 子树长度
    for i in range(n - length + 1):  # 子树起始
        j = i + length - 1  # 子树结束
        cost[i][j] = float('inf')
        for r in range(i, j + 1):  # 尝试不同的根
            c = (sum(P[i:j + 1]) + sum(Q[i:j + 2])) + (cost[i][r - 1] if r > i else 0) + (cost[r + 1][j] if r < j else 0)
            if c < cost[i][j]:
                cost[i][j] = c
                root[i][j] = r

# 输出最小成本和根节点
print("Cost Matrix:")
print(cost)
print("Root Matrix:")
print(root)



def print_tree(root, i, j, indent=""):
    if i > j:
        return
    r = root[i][j]
    print(indent + f"Node: {P[r]} (Root)")
    print_tree(root, i, r - 1, indent + "  ")
    print_tree(root, r + 1, j, indent + "  ")

print_tree(root, 0, n - 1)