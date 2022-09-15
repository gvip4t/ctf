import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load pkl data
x = pickle.load(open('inputs/X.pkl', 'rb'))
y = pickle.load(open('inputs/y.pkl', 'rb'))
x_test = pickle.load(open('inputs/X_test.pkl', 'rb'))


# Create model
# Since the goal is to fit the training data, 2 convolutional layers with no drouput is sufficient
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(80,80,3))) # Inputs are 80 x 80 RGB images
model.add(MaxPooling2D((2,2))) # Add Maxpooling layers to reduce dimensionality
model.add(Conv2D(64, (3,3), activation='relu'))

# Configure classification layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Final layer just outputs 1 bit classification

# Compile and train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
accuracy = 0
while accuracy < 0.995: # Stop training when model is fully fit
    history = model.fit(x, y, epochs=5)
    accuracy = max(history.history['accuracy'])


# Use model to classify test data
prediction = ''.join(['0' if i < 0.5 else '1' for i in model.predict(x_test)])

# Covert bytes in prediction to flag string
flag = ''
for i in range(0, len(prediction), 8):
    flag += chr(int(prediction[i:i+8], 2))
print(flag)

# Save model as h5 file
model.save('model.h5')