import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

df=pd.read_csv(r"D:\Cirrus\ml\uber.csv")

df.drop(columns=['Unnamed: 0'], inplace=True)

print(df.columns)
print(df.isna().sum())

df=df.head(1000)

print(df[df['fare_amount'] <= 0].shape)

print(df[df['passenger_count'] <= 0].shape)

print(df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].describe())

df= df[df['passenger_count'] > 0]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) *
         np.sin(dlon / 2) ** 2)

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

df['distance_km'] = haversine(
    df['pickup_latitude'],
    df['pickup_longitude'],
    df['dropoff_latitude'],
    df['dropoff_longitude']
)

print(df[["distance_km","fare_amount"]].head())

print(df)

print(df[df["distance_km"]<=0].shape)

df = df[df['distance_km'] > 0]

print(df['distance_km'].describe())

df = df[df['distance_km'] <= 70]

df = df[df['distance_km'] >= 0.1]

##building the model

x=df[["distance_km"]].values
y=df["fare_amount"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("distance_km vs fare_price")
plt.xlabel("distance_km")
plt.ylabel("fare_price")
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_test,y_pred)
print(mae)

mse=mean_squared_error(y_test,y_pred)
print(mse)

rmse=np.sqrt(mse)
print(rmse)

from sklearn.metrics import r2_score

print("R² Score:", r2_score(y_test, y_pred))

import pickle

with open("uber_fare.pkl", "wb") as file:
    pickle.dump(regressor, file)

print("Model saved successfully!")















