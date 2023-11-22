import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def new_model(X_train) -> Sequential:
    # Build the model
    # Create a MLP
    model = Sequential()
    # Use kernel_regularizer to avoid overfitting (l1l2)
    # Start with a Conv1D layer
    model.add(Conv1D(8, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1), kernel_regularizer=tf.keras.regularizers.l1(1e-5)))
    # Add a convolutional layer
    model.add(Conv1D(16, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(1e-5)))
    # Add a flatten layer
    model.add(Flatten())
    # Add one hidden layer
    model.add(Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(1e-5)))
    # Add an output layer
    model.add(Dense(1, activation='linear'))     

    print(model.summary())

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def train(data : pd.DataFrame)->Sequential:
    
    # Load your dataset
    # Assuming you have a DataFrame named 'df' with the mentioned features and the target 'quality'

    # Split the data into features and target (target is the quality column between 0 and 10)
    X = data.drop(['quality', 'Id'], axis=1)
    # Get the target
    y = data['quality']

    fBestModel = 'model/best_model.h5'

    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00001, patience=20, verbose=1, mode='auto')

    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                        verbose=0, mode='auto', min_delta=0.0001, 
                        cooldown=0, min_lr=0)

    best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

    model = new_model(X)

    # Train the model
    model.fit(X, y, epochs=200, batch_size=32, validation_split=0.2,callbacks=[best_model, early_stop, lr])
    return model

def predict(model : Sequential,X : np.ndarray):
    # Make predictions
    predictions = model.predict(X, verbose=0)
    return predictions

def accuracy(model, X_test, y_test):
    x = 0
    for i in range(X_test.shape[0]):
        # Convert the Series to a NumPy array
        X_test_np = X_test.iloc[i].values.reshape(1, -1)

        # Make predictions
        predicted_quality = round(predict(model, X_test_np)[0][0])
        actual_quality = round(y_test.iloc[i])

        if i<5:
            print("Predicted Wine Quality:", predicted_quality)
            print("Actual Wine Quality:", actual_quality)

        if predicted_quality == actual_quality:
            x += 1
    return x / len(X_test)
        
# Websocket used to communicate with the 
def websocket() :
    pass



if __name__ == '__main__':
    data = pd.read_csv('db/Wines.csv')
    model = train(data)

    # Test a prediction
    X_test = data.drop(['quality', 'Id'], axis=1)
    y_test = data['quality']

    # Get the first observation of X_test (the features) and convert the Series to a NumPy array
    X_test_np = X_test.iloc[0].values.reshape(1, -1)
    # Get the first observation of y_test (the target) and convert the Series to a NumPy array
    y_test_np = y_test.iloc[0]

    # Make predictions
    predictions = predict(model, X_test_np)
    print("Predicted Wine Quality:", round(predictions[0][0]))
    print("Actual Wine Quality:", round(y_test_np))
