import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = np.load('data_input.npy')
target = np.load('data_target.npy')

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data,target)

labels = knn.predict(data)

plt.scatter(data[:, 0], data[:, 1],c=labels)

plt.xlabel('length')
plt.ylabel('weight')
plt.show()
