import matplotlib.pyplot as plt
import cv2
import numpy as np
from Kmean_color import Kmeans
img = cv2.imread("hinh-anh-con-bo-cuoi-dep-nhat.jpg")
img_size =img.shape
X=img.reshape(img_size[0] * img_size[1], img_size[2])
km = Kmeans(n_clusters=3)
km.fit(X)
n_iter = 9
fig, ax = plt.subplots(3, 3, figsize=(16, 16))
ax = np.ravel(ax)
centers = []
for i in range(n_iter):
    centroids = km.centroids
    centers.append(centroids)
    ax[i].scatter(X[km.labels == 0, 0], X[km.labels == 0, 1],
                  c='green', label='cluster 1')
    ax[i].scatter(X[km.labels == 1, 0], X[km.labels == 1, 1],
                  c='blue', label='cluster 2')
    ax[i].scatter(X[km.labels == 2, 0], X[km.labels == 2, 1],
                  c='yellow', label='cluster 3')
    ax[i].scatter(centroids[:, 0], centroids[:, 1],
                  c='r', marker='*', s=300, label='centroid')
    ax[i].set_xlim([-201, 251])
    ax[i].set_ylim([-201, 251])
    ax[i].legend(loc='lower right')
    ax[i].set_title(f'{km.error:.4f}')
    ax[i].set_aspect('equal')
plt.tight_layout()
plt.show()