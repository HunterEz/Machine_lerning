import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

path = 'HTRU_2.csv'
df = pd.read_csv(path)

# sns.pairplot(df, hue="target", diag_kind = "kde")
# plt.show()

scaler = StandardScaler()
X = scaler.fit_transform(df.drop('target', axis=1))
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Точность:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Accuracy: 0.9810055865921787
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.99      0.99      0.99      3259
#            1       0.93      0.85      0.89       321

#     accuracy                           0.98      3580
#    macro avg       0.96      0.92      0.94      3580
# weighted avg       0.98      0.98      0.98      3580

# Задчи решаемые в космических маштабах, требуют максимальную точность, значение в 98% кажется недостаточным
# Однако не в исследовательских целях точность достаточно высока и является удовлетворительной в других классах задач: учебная, демонстарционная и тд.