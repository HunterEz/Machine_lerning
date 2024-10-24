from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
import random

from sklearn.svm import SVC

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import SpectralClustering

import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
X = iris.data
y = iris.target

# классификация набора алгоритмом k-ближайших соседей.
# минимизировать инерцию, выбирая такие центры кластеров, 
# для которых сумма квадратов расстояний от центра до точек минимальна
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y)
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 3], hue=y_pred_knn, marker='x')
plt.title('k-блишайших соседей')
plt.show()

# классификация набора алгоритмом Случайный лес.
# подгоняет ряд классификаторов дерева решений к различным подвыборкам набора данных и использует 
# усреднение для повышения точности прогнозирования и контроля переобучения
rf = RandomForestClassifier(n_estimators=100, random_state=random.randint(0, 100))
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y)
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 3], hue=y_pred_rf, marker='x')
plt.title('Случайный лес')
plt.show()

# классификация набора машинами опорных векторов (SVM)
svm = SVC(kernel='linear', random_state=random.randint(0, 100))
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y)
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 3], hue=y_pred_svm, marker='x')
plt.title('SVM')
plt.show()

# кластеризация алгоритмом k-средних
# кластеризует данные, пытаясь разделить образцы на n групп с одинаковой дисперсией, 
# минимизируя критерий, известный как инерция или сумма квадратов внутри кластера
kmeans = KMeans(n_clusters=3, random_state=random.randint(0, 100))
kmeans.fit(X)

y_pred_kmeans = kmeans.predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y_pred_kmeans)
plt.title('k-средних')
plt.show()

# иерархическая кластеризация методом Уорда
X_scaled = (X - X.min()) / (X.max() - X.min())

linkage_matrix = ward(X_scaled)

plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title('Метод Уорда')
plt.show()

# спектральная кластеризация
sc = SpectralClustering(n_clusters=3, random_state=42)
y_pred_spectral = sc.fit_predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y_pred_spectral)
plt.title('Спектральная кластеризация')
plt.show()