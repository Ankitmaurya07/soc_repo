import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error
import math

def get_train_test(url, split_ratio=0.8):
    data = pd.read_csv(url, header=0)
    x = data.iloc[:, 1].values  # Assuming sunspot values are in the second column

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x.reshape(-1, 1)).flatten()

    split_point = int(len(x) * split_ratio)

    train_data = x[:split_point]
    test_data = x[split_point:]

    return train_data, test_data, scaler

sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
train_data, test_data, scaler = get_train_test(sunspots_url)

# Prepare the data for RNN
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step)]
        X.append(a)
        Y.append(dataset[i + time_step])
    return np.array(X), np.array(Y)

time_step = 1
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape inputs for RNN [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(64, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate RMSE
train_error = math.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
test_error = math.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))

print('Train RMSE:', train_error)
print('Test RMSE:', test_error)

# Visualize the results
actual = np.append(y_train[0], y_test[0])
predictions = np.append(train_predict[:,0], test_predict[:,0])
rows = len(actual)

plt.figure(figsize=(15, 6), dpi=120)
plt.plot(range(rows), actual)
plt.plot(range(rows), predictions)
plt.axvline(x=len(y_train[0]), color='r')
plt.legend(['Actual', 'Predictions'])
plt.xlabel('Observation number after given time steps')
plt.ylabel('Sunspots scaled')
plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
plt.show()
