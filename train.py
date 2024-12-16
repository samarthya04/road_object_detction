import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
import numpy as np
import os

# Configuration
IMAGE_SIZE = 240
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 27  # Will be set based on dataset

def create_model(num_classes):
    # Load MobileNetV2 as base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Add detection specific layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    
    # Output layers for object detection
    classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification')(x)
    bounding_box_output = tf.keras.layers.Dense(4, name='bounding_box')(x)  # (x, y, width, height)
    
    model = tf.keras.Model(inputs=inputs, outputs=[classification_output, bounding_box_output])
    return model

def prepare_dataset(data_dir):
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

def main():
    # Set up data directory
    data_dir = 'path/to/your/dataset'  # Update this with your dataset path
    
    # Prepare dataset
    train_generator, validation_generator = prepare_dataset(data_dir)
    global NUM_CLASSES
    NUM_CLASSES = len(train_generator.class_indices)
    
    # Create and compile model
    model = create_model(NUM_CLASSES)
    
    # Compile with appropriate losses
    losses = {
        'classification': 'categorical_crossentropy',
        'bounding_box': 'mse'
    }
    
    loss_weights = {
        'classification': 1.0,
        'bounding_box': 1.0
    }
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=losses,
        loss_weights=loss_weights,
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open('object_detection_model.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    main()
