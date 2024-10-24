import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

# @brief read_csv Считать файл значений, разделенных запятыми (csv), в DataFrame
weather = pd.read_csv('weatherHistory.csv') 

weather['Formatted Date'] = pd.to_datetime(weather['Formatted Date'], utc=True)
weather.Summary = weather.Summary.astype('category')
weather['Precip Type'] = weather['Precip Type'].astype('category')
weather['Daily Summary'] = weather['Daily Summary'].astype('category')
weather.info()

# weather.info() result:

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 96453 entries, 0 to 96452
# Data columns (total 12 columns):
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Formatted Date            96453 non-null  datetime64[ns, UTC]
#  1   Summary                   96453 non-null  category
#  2   Precip Type               95936 non-null  category
#  3   Temperature (C)           96453 non-null  float64
#  4   Apparent Temperature (C)  96453 non-null  float64
#  5   Humidity                  96453 non-null  float64
#  6   Wind Speed (km/h)         96453 non-null  float64
#  7   Wind Bearing (degrees)    96453 non-null  float64
#  8   Visibility (km)           96453 non-null  float64
#  9   Loud Cover                96453 non-null  float64
#  10  Pressure (millibars)      96453 non-null  float64
#  11  Daily Summary             96453 non-null  category
# dtypes: category(3), datetime64[ns, UTC](1), float64(8)
# memory usage: 7.0 MB

# @brief train_test_split Разбить массивы или матрицы на случайные обучающие и тестовые подмножества
# @param test_size - долю набора данных для включения в тестовое разделениеl
#        random_state - управляет перемешиванием, применяемым к данным перед применением разделения
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(weather['Humidity'], weather['Apparent Temperature (C)'], test_size = 0.33, random_state = 42)

# Диграмма рассеяния для тренировочныъ данных
sns.regplot(x = x_train, y = y_train)
plt.show()

model = LinearRegression()
#@brief fit подходящая линейная модель
model.fit(x_train.to_frame(), y_train.to_frame())
#@brief predict прогнозирование с использованием линейной модели
temp_predict = model.predict(x_test.to_frame())

# Диграммы рассеяния для предсказаний по показателям влажности
sns.regplot(x = x_test, y = temp_predict)
plt.show()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(weather[['Humidity', 'Wind Speed (km/h)']], weather['Apparent Temperature (C)'], test_size=0.33, random_state = 42)

# Диграммы рассеяния для тренировочныъ данных
sns.regplot(x = x_train['Humidity'], y = y_train)
plt.show()
sns.regplot(x = x_train['Wind Speed (km/h)'], y = y_train)
plt.show()

model.fit(x_train, y_train.to_frame())
temp_predict = model.predict(x_test)

# Диграммы рассеяния для предсказаний по показателям влажности (2d проекция)
sns.regplot(x = x_test['Humidity'], y = temp_predict)
plt.show()

# Диграммы рассеяния для предсказаний по показателям скорости ветра (2d проекция)
sns.regplot(x = x_test['Wind Speed (km/h)'], y = temp_predict)
plt.show()

#3d модель предсказаний по влажности и скорости ветра
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = x_test['Humidity']
y = x_test['Wind Speed (km/h)']
z = temp_predict

ax.scatter(x, y, z)
ax.set_xlabel("Влажность")
ax.set_ylabel("Скорость ветра")
ax.set_zlabel("Ощущаемая температура")
plt.show()