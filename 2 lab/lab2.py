import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# @brief метод главных компонент
# @param X - входные данные; k - искомое число главных компонент
# @return X_res набор данных меньшей размерности
def PCA_realization(X, k):
    normal_X = X - X.mean(axis=0) # Нормализация данных 
    covMatrix = np.cov(normal_X.T) # Построение матрицы ковариации
    diagVal, diagVectors = np.linalg.eig(covMatrix) # Диагонализация матрицы
    
    # Сортировка значений векторов
    index = np.argsort(diagVal)[::-1]
    diagVal = diagVal[index]
    diagVectors = diagVectors[:, index]
    
    diagVectors = diagVectors[:, :k] # Оставление только k наибольших векторов
    X_res = np.dot(normal_X, diagVectors) # Модификация исходных данных
    
    return X_res

if __name__ == '__main__':
    #data load
    data_full, target = load_breast_cancer(return_X_y=True)
    data = data_full[:, :30]

    #PCA start
    pca_sklearn = PCA(n_components=30)
    pca_sklearn_val = pca_sklearn.fit_transform(data)
    my_pca_val = PCA_realization(data, k=30)

    #result
    print(my_pca_val)
# [[ 1.16014257e+03 -2.93917544e+02  4.85783976e+01 ...  1.29334919e-03
#   -1.98910417e-03  7.04378359e-04]
#  [ 1.26912244e+03  1.56301818e+01 -3.53945342e+01 ... -1.34685217e-03
#   -6.85925212e-04 -1.06125086e-03]
#  [ 9.95793889e+02  3.91567432e+01 -1.70975298e+00 ...  1.84867758e-05
#    7.75218581e-04  4.05360270e-04]
#  ...
#  [ 3.14501756e+02  4.75535252e+01 -1.04424072e+01 ...  2.54369639e-05
#   -4.83858890e-04 -2.85342703e-04]
#  [ 1.12485812e+03  3.41292250e+01 -1.97420874e+01 ...  1.23547951e-03
#    8.08728730e-04  1.21655195e-03]
#  [-7.71527622e+02 -8.86431064e+01  2.38890319e+01 ... -4.44552928e-03
#   -2.42876427e-04  1.46800350e-03]]
    print(pca_sklearn_val)
# [[ 1.16014257e+03 -2.93917544e+02  4.85783976e+01 ...  1.29334920e-03
#    1.98910417e-03  7.04378357e-04]
#  [ 1.26912244e+03  1.56301818e+01 -3.53945342e+01 ... -1.34685217e-03
#    6.85925212e-04 -1.06125086e-03]
#  [ 9.95793889e+02  3.91567432e+01 -1.70975298e+00 ...  1.84867755e-05
#   -7.75218581e-04  4.05360270e-04]
#  ...
#  [ 3.14501756e+02  4.75535252e+01 -1.04424072e+01 ...  2.54369636e-05
#    4.83858890e-04 -2.85342702e-04]
#  [ 1.12485812e+03  3.41292250e+01 -1.97420874e+01 ...  1.23547951e-03
#   -8.08728729e-04  1.21655195e-03]
#  [-7.71527622e+02 -8.86431064e+01  2.38890319e+01 ... -4.44552928e-03
#    2.42876427e-04  1.46800350e-03]]

    #visualization
    sklearn_expl_var_ratio = pca_sklearn.explained_variance_ratio_
    my_expl_var_ratio = np.var(my_pca_val, axis=0) / np.var(data, axis=0).sum()
    plt.plot(range(1, 5), sklearn_expl_var_ratio[0:4], label='PCA from Scikit-learn')
    plt.plot(range(1, 5), my_expl_var_ratio[0:4], "--", label='PCA realization')
    plt.legend()
    plt.xlabel('Components')
    plt.ylabel('Eigenvalues')
    plt.grid()
    plt.show()