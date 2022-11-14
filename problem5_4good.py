import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.svm import SVC
import numpy as np


def mesh_dataset(dataset):
    vert_tol = 3
    hor_tol = 3

    x_1 = dataset[:, 0]
    x_2 = dataset[:, 1]

    x_1, x_2 = np.meshgrid(np.linspace(
        x_1.min()-hor_tol, x_1.max()+hor_tol, 20), np.linspace(x_2.min()-vert_tol, x_2.max()+vert_tol, 20))

    return [x_1, x_2]


def classify_moons(dataset, labels, kernel_choice, C_choice, cmap_choice, gamma_choice=1):
    mesh = mesh_dataset(dataset)

    xi = mesh[0].ravel()
    yi = mesh[1].ravel()

    cmap = plt.get_cmap(cmap_choice, 3)

    if kernel_choice == "linear":
        clf = SVC(kernel="linear", C=C_choice)
    elif kernel_choice == "rbf":
        clf = SVC(kernel="rbf", C=C_choice, gamma=gamma_choice)

    clf.fit(dataset, labels)
    return clf.predict(np.c_[xi, yi])


plt.figure(figsize=(5, 5))
plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)


dataset, labels = make_moons(1000, noise=0.1)
cmap_choice = "jet"

mesh_plot_size = 1
data_plot_size = 20

x1_data = dataset[:, 0]
x2_data = dataset[:, 1]

mesh = mesh_dataset(dataset)

xi_mesh = mesh[0]
xi_raveled = xi_mesh.ravel()
yi_mesh = mesh[1]
yi_raveled = yi_mesh.ravel()

r_choice = [1, 1, 1e-2, 1e5]
gamma_choice = [None, 1, 1e-5, 1e5]

plt.subplot(221)
pred = classify_moons(dataset, labels, "linear", r_choice[0], cmap_choice)

plt.scatter(xi_raveled, yi_raveled, c=pred, s=mesh_plot_size, cmap=cmap_choice)
plt.scatter(x1_data, x2_data, c=labels, s=data_plot_size,
            cmap=cmap_choice, ec='black')
plt.contourf(xi_mesh, yi_mesh, pred.reshape(
    xi_mesh.shape), cmap=cmap_choice, alpha=0.5, levels=1, zorder=0)
plt.title("Linear kernel, r = " + str(r_choice[0]))

plt.subplot(222)
cmap_choice = "plasma"

pred = classify_moons(dataset, labels, "rbf", 1.0, cmap_choice, 1)
plt.scatter(xi_raveled, yi_raveled, c=pred, s=mesh_plot_size, cmap=cmap_choice)
plt.scatter(x1_data, x2_data, c=labels, s=data_plot_size,
            cmap=cmap_choice, ec='black')
plt.contourf(xi_mesh, yi_mesh, pred.reshape(
    xi_mesh.shape), cmap=cmap_choice, alpha=0.6, levels=1, zorder=0)
plt.title("REASONABLE - Radial basis function kernel, r = " +
          str(r_choice[1]) + ", gamma = " + str(gamma_choice[1]))

plt.subplot(223)

pred = classify_moons(dataset, labels, "rbf", .01, cmap_choice, .0001)
plt.scatter(xi_raveled, yi_raveled, c=pred, s=mesh_plot_size, cmap=cmap_choice)
plt.scatter(x1_data, x2_data, c=labels, s=data_plot_size,
            cmap=cmap_choice, ec='black')
plt.contourf(xi_mesh, yi_mesh, pred.reshape(
    xi_mesh.shape), cmap=cmap_choice, alpha=0.6, levels=1, zorder=0)
plt.title("UNDERFIT - Radial basis function kernel, r = " +
          str(r_choice[2]) + ", gamma = " + str(gamma_choice[2]))
plt.subplot(224)

pred = classify_moons(dataset, labels, "rbf", 50, cmap_choice, 50)
plt.scatter(xi_raveled, yi_raveled, c=pred, s=mesh_plot_size, cmap=cmap_choice)
plt.scatter(x1_data, x2_data, c=labels, s=data_plot_size,
            cmap=cmap_choice, ec='black')
plt.contourf(xi_mesh, yi_mesh, pred.reshape(
    xi_mesh.shape), cmap=cmap_choice, alpha=0.6, levels=1, zorder=0)
plt.title("OVERFIT - Radial basis function kernel, r = " +
          str(r_choice[3]) + ", gamma = " + str(gamma_choice[3]))

plt.show()
