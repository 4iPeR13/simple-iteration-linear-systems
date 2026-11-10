import numpy as np


A = np.array([
    [3.81, 0.25, 1.28, 0.75],
    [2.25, 1.32, 4.58, 0.49],
    [5.31, 6.28, 0.98, 1.04],
    [9.39, 2.45, 3.35, 2.38]
])

b = np.array([4.21, 6.47, 2.38, 10.48])

n = len(b)
eps = 1e-4
x = np.zeros(n)  # Початкове наближення x^(0)
max_iterations = 1000

C = np.zeros_like(A)
d = np.zeros_like(b)

for i in range(n):
    for j in range(n):
        if i != j:
            C[i][j] = -A[i][j] / A[i][i]
    d[i] = b[i] / A[i][i]


q = max(np.sum(np.abs(C), axis=1))

# Ітераційний процес: x^(k+1) = C * x^(k) + d
print(f"q = {q:.6f}")
for k in range(max_iterations):
    x_new = C @ x + d
    norm_diff = np.max(np.abs(x_new - x))
    criterion = (1 / (1 - q)) * norm_diff

    # Вивід на кожній ітерації
    print(f"\nІтерація {k + 1}")
    for i in range(n):
        print(f"x[{i+1}] = {x_new[i]:.6f}")
    print(f"Критерій зупинки: {criterion:.6e}")

    if criterion < eps:
        print("\nЗбіжність досягнута.")
        break
    x = x_new
else:
    print("\nБез збіжності.")

# Вектор нев’язки
residual = np.abs(b - A @ x)
print("\nВектор нев’язки:")
for i in range(n):
    print(f"r[{i+1}] = {residual[i]:.6e}")
