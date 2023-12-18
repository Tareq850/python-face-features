import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization

# Determine the size of images
img_height, img_width = 64, 64

# Specify the data generator and data folder path
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
data_dir = 'C:/Users/UX/.spyder-py3/train'
batch_size = 32
img_height, img_width = 64, 64
# Define batch data
batch_size = 32

# Define folders for training and validation
train_data = data_generator.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Define the multiple classification format
    subset='training'
)

validation_data = data_generator.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Define the multiple classification format
    subset='validation'
)
# Building a machine learning model
num_classes = 7  # Replace this with your number of categories
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    
    Dense(512),
    LeakyReLU(alpha=0.1),  # Use Leaky ReLU 
    
    Dropout(0.5),  # Add Dropout to avoid overload
    
    BatchNormalization(),  # Add Batch Normalization
    
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert category labels to one-hot encoding
train_labels = to_categorical(train_data.classes)
validation_labels = to_categorical(validation_data.classes)

# Model training
epochs = 10
history = model.fit(train_data, epochs=epochs, validation_data=validation_data)

model.save('smile.keras')
