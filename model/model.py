import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curves(history):
    
    plt.figure(figsize=(10,5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('#epoch')
    plt.legend(['Training dataset', 'Testing dataset'], loc='upper left')
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('#epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Load your dataset
# Assuming you have a DataFrame named 'df' with the mentioned features and the target 'quality'
df = pd.read_csv('db/Wines.csv')

# Split the data into features and target (target is the quality column between 0 and 10)
X = df.drop(['quality', 'Id'], axis=1)
# Convert the target to a normalized output between 0 and 1 (because quality is between 0 and 10)
y = (df['quality']-min(df['quality'])) / (max(df['quality']) - min(df['quality']))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))  # Hidden layer with 64 neurons and ReLU activation
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons and ReLU activation
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron and sigmoid activation

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Model accuracy (on test dataset): {0}%".format(round(accuracy * 100,2)))

# Make predictions
predictions = model.predict(X_test_scaled)

