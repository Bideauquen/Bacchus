import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
# Assuming you have a DataFrame named 'df' with the mentioned features and the target 'quality'
df = pd.read_csv('db/Wines.csv')

# Split the data into features and target (target is the quality column between 0 and 10)
X = df.drop(['quality', 'Id'], axis=1)
# Convert the target to a normalized output between 0 and 1 (because quality is between 0 and 10)
y = (df['quality']-min(df['quality'])) / (max(df['quality']) - min(df['quality']))

def new_model(X_train) -> Sequential:
    # Build the model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Hidden layer with 64 neurons and ReLU activation
    model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons and ReLU activation
    model.add(Dense(1))  # Output layer with 1 neuron and sigmoid activation    

    print(model.summary())

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def train(X,y)->Sequential:
    fBestModel = 'model/best_model.h5'

    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00001, patience=10, verbose=1, mode='auto')

    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, 
                        verbose=0, mode='auto', min_delta=0.0001, 
                        cooldown=0, min_lr=0)

    best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)


    # Standardize the features (optional but recommended for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)

    model = new_model(X_train_scaled)

    
    # Train the model
    model.fit(X_train_scaled, y, epochs=200, batch_size=32, validation_split=0.2,callbacks=[best_model, early_stop])
    return model

def predict(model,X_test_scaled):
    # Make predictions
    predictions = model.predict(X_test_scaled, verbose=0)
    return predictions * (max(df['quality']) - min(df['quality'])) + min(df['quality'])

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
            print("Actual Wine Quality:", actual_quality * (max(df['quality']) - min(df['quality'])) + min(df['quality']))

        if predicted_quality == actual_quality:
            x += 1
    return x / len(X_test)
        


if __name__ == '__main__':
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = new_model(X_train)
    model.load_weights('model/best_model.h5')
    print(accuracy(model,X_test,y_test))