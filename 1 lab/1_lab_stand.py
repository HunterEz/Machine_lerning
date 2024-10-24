import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn

weather = pd.read_csv('weatherHistory.csv') 

weather['Formatted Date'] = pd.to_datetime(weather['Formatted Date'], utc=True)
weather.Summary = weather.Summary.astype('category')
weather['Precip Type'] = weather['Precip Type'].astype('category')
weather['Daily Summary'] = weather['Daily Summary'].astype('category')

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(weather['Humidity'], weather['Apparent Temperature (C)'], test_size = 0.33, random_state = 42)

model = LinearRegression()
model.fit(x_train.to_frame(), y_train.to_frame())

x_train2, x_test2, y_train2, y_test2 = sklearn.model_selection.train_test_split(weather[['Humidity', 'Wind Speed (km/h)']], weather['Apparent Temperature (C)'], test_size=0.33, random_state = 42)
model2 = LinearRegression()
model2.fit(x_train2, y_train2.to_frame())

print("Enter value \nHumidity(0-1) Wind Speed (km/h) > 0 \nFormat: H WS(optional)")

while(True):
    
    my_list = [float(el) for el in input("\n>>").split()]

    if(len(my_list) == 1):
        if(my_list[0] > 1 or my_list[0] < 0):
            print("\nError: incorrect data")
        else:
            print("\nHumidity = " + str(my_list[0]) + "\tApparent Temperature: " + str(model.predict([[my_list[0]]])))
    elif(len(my_list) == 2):
        if((my_list[0] > 1 or my_list[0] < 0) or (my_list[1] < 0 or my_list[1] > 410)):
            print("\nError: incorrect data")
        else:
            print("\nHumidity = " + str(my_list[0]) + ", Wind Speed = " + str(my_list[1]) + "\tApparent Temperature: " + str(model2.predict([[my_list[0], my_list[1]]])))
