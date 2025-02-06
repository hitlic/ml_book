import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt


x, y = np.mgrid[-2:5:.1, -2:5:.1]
pos = np.dstack((x, y))

norm1 = multivariate_normal([0.4, 1.2], [[1.0, 0.1], [0.7, 0.7]])
norm2 = multivariate_normal([2.5, 1.5], [[2.0, -0.7], [-0.9, 0.5]])
z = 0.5 * norm1.pdf(pos) + 0.5 * norm2.pdf(pos)

fig = plt.figure(figsize=[12, 5])

ax = fig.add_subplot(121)
ax.contourf(x, y, z, cmap='coolwarm')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='coolwarm')
ax.view_init(elev=65, azim=260)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# plt.savefig('fig.png', dpi=300)
plt.show()
