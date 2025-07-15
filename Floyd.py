import numpy as np

matrix_data = [
    [0, 194206, float('-inf'), 118341, 116256, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    [float('-inf'), 0, 85560, float('-inf'), 86256, 83195, float('-inf'), float('-inf'), float('-inf')],
    [float('-inf'), float('-inf'), 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'),
     float('-inf')],
    [float('-inf'), float('-inf'), float('-inf'), 0, 10391, float('-inf'), 11341, 8509, float('-inf')],
    [float('-inf'), float('-inf'), float('-inf'), float('-inf'), 0, 5245, float('-inf'), 6424, 5032],
    [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), 0, float('-inf'), float('-inf'),
     float('-inf')],
    [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), 0, float('-inf'),
     float('-inf')],
    [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), 0,
     float('-inf')],
    [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'),
     float('-inf'), 0]
]

labels = ["(0,0)", "(0,1)", "(0,2)", "(1,0)", "(1,1)", "(1,2)", "(2,0)", "(2,1)", "(2,2)"]

n = len(matrix_data)
dist = np.array(matrix_data, dtype=float)
pred = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        if i == j:
            pred[i][j] = i
        elif not np.isinf(dist[i][j]):
            pred[i][j] = i
        else:
            pred[i][j] = -1

for k in range(n):
    for i in range(n):
        for j in range(n):
            if dist[i][k] != float('-inf') and dist[k][j] != float('-inf'):
                if dist[i][j] < dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]

print("最长路径距离矩阵:")
for i in range(n):
    row = []
    for j in range(n):
        if dist[i][j] == float('-inf'):
            row.append("-∞")
        else:
            row.append(str(int(dist[i][j])))
    print("\t".join(row))

start = 0
end = 8
if dist[start][end] == float('-inf'):
    print(f"There is no path from {labels[start]} to {labels[end]}")
else:
    print(f"\nThe longest path distance from {labels[start]} to {labels[end]}: {int(dist[start][end])}"  )


    path = []
    current = end
    while current != start:
        path.append(current)
        current = pred[start][current]
    path.append(start)
    path.reverse()

    print("path:", " -> ".join([labels[i] for i in path]))