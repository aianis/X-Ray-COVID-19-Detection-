from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import * 
from tensorflow.keras.utils import plot_model
import os 
os.system("cls") # Clear the terminal 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # To avoid the error: "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."

# model architecture 
model = Sequential() 
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu")) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))

# Adding additional layers for increased complexity
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu")) 
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu")) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu")) 
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu")) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid")) 

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary() 

train_datagen = image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = image.ImageDataGenerator(rescale=1./255) 
print("Created the data generator object.")

# Loading the dataset 
train_generator = train_datagen.flow_from_directory('CovidDataset/Train',target_size=(224,224),batch_size=32, class_mode="binary")
val_generator = test_datagen.flow_from_directory('CovidDataset/Test',target_size=(224,224),batch_size=32, class_mode="binary")

# Training the model 
hist = model.fit(train_generator, epochs=6, validation_data=val_generator, validation_steps=2)


# Saving the model
model.save('my_model.h5') # This will save your model in a file called my_model.h5
print("Model saved successfully.")