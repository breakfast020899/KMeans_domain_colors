import cv2
import matplotlib.pyplot as plt
from Kmean_color import Kmeans
def figure_SSE():
    img = cv2.imread("images.jpeg")
    img_size =img.shape
    X=img.reshape(img_size[0] * img_size[1], img_size[2])
    sse = []
    list_k = list(range(1, 10))

    for k in list_k:
        km = Kmeans(n_clusters=k)
        km.fit(X)
        sse.append(km.compute_sse(X,labels=km.labels,centroids=km.centroids))

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance');
    plt.show()
#figure_color()
