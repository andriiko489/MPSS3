import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Параметри для варіанту 6 (вода)
a = 0.143e-6      # м^2/с
L = 0.75          # м
T_hours = 10      # годин
N = 100           # кількість шарів
h = 0.3 * 3600    # крок по часу у секундах (0.3 год)

# Граничні умови
alpha = 6         # зліва (°C)
beta = 25         # справа (°C)
phi_y = 0         # початкова температура (°C)

# Часові параметри
T = T_hours * 3600      # загальний час у секундах
M = int(T / h)          # кількість часових кроків
delta = L / (N + 1)     # товщина одного шару
mu = a / (delta**2)     # коефіцієнт в ОДУ

# Початкові значення температури
u = np.zeros(N)
# Результат для візуалізації
result = np.zeros((M, N))
result[0] = u.copy()

# Метод Рунге-Кутта 4-го порядку
def runge_kutta_step(u):
    def f(u):
        du = np.zeros_like(u)
        for i in range(N):
            u_left = alpha if i == 0 else u[i - 1]
            u_right = beta if i == N - 1 else u[i + 1]
            du[i] = mu * (u_right - 2 * u[i] + u_left)
        return du

    k1 = f(u)
    k2 = f(u + h * k1 / 2)
    k3 = f(u + h * k2 / 2)
    k4 = f(u + h * k3)
    return u + h * (k1 + 2*k2 + 2*k3 + k4) / 6

# Основний цикл інтегрування
for step in range(1, M):
    u = runge_kutta_step(u)
    result[step] = u.copy()


def analytical_solution(t, y, terms=30):
    series = 0
    for n in range(1, terms + 1):
        term = ((1 - (-1)**n) / (n * np.pi)) * \
               np.exp(-a * (n * np.pi / L)**2 * t) * \
               np.sin(n * np.pi * y / L)
        series += term
    return (alpha + beta) / 2 - (beta - alpha) * series



ys = np.linspace(0, L, N)
ts = np.linspace(0, T, M)
T_grid, Y_grid = np.meshgrid(ts / 3600, ys)

# Чисельне рішення
U_numeric = result.T

# Аналітичне рішення
U_exact = np.zeros_like(U_numeric)
for i, y in enumerate(ys):
    for j, t in enumerate(ts):
        U_exact[i, j] = analytical_solution(t, y)

# MAE & MSE
MAE = np.max(np.abs(U_numeric - U_exact))
MSE = np.mean((U_numeric - U_exact) ** 2)
print(f"MAE: {MAE:.6f} °C")
print(f"MSE: {MSE:.6f} °C²")

# Побудова графіків
fig = plt.figure(figsize=(14, 6))

# Числовий розв'язок
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(T_grid, Y_grid, U_numeric, cmap=cm.viridis)
ax1.set_title('Чисельне рішення (метод Рунге-Кутта)')
ax1.set_xlabel('Час (год)')
ax1.set_ylabel('Товщина стінки (м)')
ax1.set_zlabel('Температура (°C)')

# Аналітичне рішення
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T_grid, Y_grid, U_exact, cmap=cm.plasma)
ax2.set_title('Аналітичне рішення')
ax2.set_xlabel('Час (год)')
ax2.set_ylabel('Товщина стінки (м)')
ax2.set_zlabel('Температура (°C)')

plt.tight_layout()
plt.show()
