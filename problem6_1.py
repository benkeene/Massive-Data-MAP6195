import sklearn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

d = load_digits()

filtered_data = []
filtered_images = []
filtered_target = []

for i in range(len(d.target)):
    if d.target[i] == 1 or d.target[i] == 5:
        filtered_data.append(d.data[i])
        filtered_images.append(d.images[i])
        filtered_target.append(d.target[i])

pca = PCA(2)
data_pca = pca.fit_transform(filtered_data)

filtered_data = np.array(filtered_data)
filtered_images = np.array(filtered_images)
filtered_target = np.array(filtered_target)

pca_1x = data_pca[:, 0][filtered_target == 1]
pca_1y = data_pca[:, 1][filtered_target == 1]

pca_5x = data_pca[:, 0][filtered_target == 5]
pca_5y = data_pca[:, 1][filtered_target == 5]

plt.close()

fig = plt.figure()

avg_1 = np.zeros([8, 8])
avg_5 = np.zeros([8, 8])

for i in range(len(filtered_target)):
    if filtered_target[i] == 1:
        avg_1 += filtered_images[i]
    else:
        avg_5 += filtered_images[i]

avg_1 /= len(filtered_target)
avg_5 /= len(filtered_target)

idx = np.random.choice(np.arange(len(pca_1x)), 100)

ax1 = plt.subplot(211)
ax1.scatter(pca_1x[idx], pca_1y[idx], c='blue', label=1)
ax1.scatter(pca_5x[idx], pca_5y[idx], c='red', label=5)
ax1.legend()
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('2-dim PCA of 1s and 5s, random sampling of 100 of each')

ax2 = plt.subplot(223)
ax2.matshow(avg_1)
ax2.set_title('pixelwise averaged 1')

ax3 = plt.subplot(224)
ax3.matshow(avg_5)
ax3.set_title('pixelwise averaged 5')

plt.tight_layout()

plt.show()
