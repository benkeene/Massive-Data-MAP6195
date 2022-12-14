""" In this problem, we test SVMs with and without kernel.
    We use the library scikit-learn (available in a standard Anaconda install).

    Generate your data with sklearn.datasets.make_moons.
    You will select some appropriate parameters below.

    Use SVCLinks to an external site. with kernel='linear' or 
    alternatively SGDClassifierLinks to an external site. to classify the data.

    Use SVCLinks to an external site. with kernel='rbf' to classify the data.
    A detailed mathematical description of the method can be found at SVCLinks to an external site..
    Make one figure for each classifier with

    matplotlib.pyplot.contourf for the classifier (you can lower the transparency with parameter alpha)
    matplotlib.pyplot.scatter for the training data.

    For all problems above choose suitable parameters for data and method.
    Use one single parameter set for the linear classifier and three different ones for the rbf kernel: One instance with under-fitting, one with over-fitting and one with a good prediction.
 """
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.svm import SVC
import numpy as np


plt.figure(figsize=(1,2))
plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)


#plt.subplot(321)
#plt.title("Moons data", fontsize="small")
X1, Y1 = make_moons(10000)
#plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

plt.subplot(321)
clf = SVC(kernel='linear', C = 1.0)
clf.fit(X1, Y1)

x2, y2 = np.meshgrid(np.linspace(X1[:, 0].min()-.5, X1[:, 0].max()+.5, 20),
                     np.linspace(X1[:, 1].min()-.5, X1[:, 1].max()+.5, 20) )

pred = clf.predict(np.c_[x2.ravel(), y2.ravel()])

cmap = plt.get_cmap('Set1', 3)

plt.scatter(x2.ravel(), y2.ravel(), c=pred, s=10, cmap=cmap, label='Linear prediction on grid')

plt.scatter(X1[:, 0], X1[:, 1], c=Y1, s=50, cmap=cmap, ec='black', label='Given values')

plt.contourf(x2, y2, pred.reshape(x2.shape), cmap=cmap, alpha=0.6, levels=2, zorder=0)

plt.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5,1.01))

plt.subplot(322)
clf = SVC(kernel='rbf', C = 1)
clf.fit(X1, Y1)

x2, y2 = np.meshgrid(np.linspace(X1[:, 0].min()-.5, X1[:, 0].max()+.5, 20),
                     np.linspace(X1[:, 1].min()-.5, X1[:, 1].max()+.5, 20) )

pred = clf.predict(np.c_[x2.ravel(), y2.ravel()])

cmap = plt.get_cmap('Set1', 3)

plt.scatter(x2.ravel(), y2.ravel(), c=pred, s=10, cmap=cmap, label='rbf prediction on grid')

plt.scatter(X1[:, 0], X1[:, 1], c=Y1, s=50, cmap=cmap, ec='black', label='Given values')

plt.contourf(x2, y2, pred.reshape(x2.shape), cmap=cmap, alpha=0.6, levels=2, zorder=0)

plt.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5,1.01))

plt.show()