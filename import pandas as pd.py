import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('weather_data.csv')
df.fillna(df.mean(), inplace=True)

df['Year'] = pd.to_datetime(df['date']).dt.year

print(df.describe())
sns.lineplot(x='Year', y='temperature', data=df)
plt.title('Temperature Trends Over Years')
plt.show()

sns.barplot(x='Year', y='rainfall', data=df)
plt.title('Yearly Rainfall Distribution')
plt.show()

sns.scatterplot(x='temperature', y='humidity', data=df)
plt.title('Humidity vs Temperature')
plt.show()

X = df[['Year']]
y = df['temperature']
model = LinearRegression()
model.fit(X, y)
df['Predicted_Temperature'] = model.predict(X)

plt.plot(df['Year'], df['temperature'], label='Actual')
plt.plot(df['Year'], df['Predicted_Temperature'], label='Predicted', color='orange')
plt.legend()
plt.title('Temperature Prediction for Next Years')
plt.show()

mse = mean_squared_error(y, df['Predicted_Temperature'])
rmse = np.sqrt(mse)
print(f'MSE: {mse}, RMSE: {rmse}')
