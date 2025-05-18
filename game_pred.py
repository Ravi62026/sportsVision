import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
import os
import numpy as np
import math

# Set image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Use only 10% of data for faster training
SUBSET_FRACTION = 0.1

# Data directories
train_dir = os.path.join('data', 'train')
valid_dir = os.path.join('data', 'valid')
test_dir = os.path.join('data', 'test')

# Data augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

# Only preprocessing for validation and testing
valid_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Calculate the number of samples to use (10% of the data)
train_sample_count = int(train_generator.samples * SUBSET_FRACTION)
valid_sample_count = int(validation_generator.samples * SUBSET_FRACTION)
test_sample_count = int(test_generator.samples * SUBSET_FRACTION)

# Calculate steps per epoch based on reduced sample count
train_steps = math.ceil(train_sample_count / BATCH_SIZE)
valid_steps = math.ceil(valid_sample_count / BATCH_SIZE)
test_steps = math.ceil(test_sample_count / BATCH_SIZE)

# Ensure we have at least one step
train_steps = max(train_steps, 1)
valid_steps = max(valid_steps, 1)
test_steps = max(test_steps, 1)

print(f"Using {SUBSET_FRACTION*100}% of data:")
print(f"Training samples: {train_sample_count} (from {train_generator.samples})")
print(f"Validation samples: {valid_sample_count} (from {validation_generator.samples})")
print(f"Test samples: {test_sample_count} (from {test_generator.samples})")
print(f"Training steps: {train_steps}, Validation steps: {valid_steps}, Test steps: {test_steps}")

num_classes = len(train_generator.class_indices)

def create_model():
    # Load the pretrained ResNet50V2 model
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Fine-tune from this layer onwards - reducing fine-tune layers for faster training
    fine_tune_at = 150  # Freeze more layers
    
    # Freeze layers before the fine_tune_at layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = models.Sequential([
        # Base model
        base_model,
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Simplified layers for faster training
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create and compile the model
model = create_model()

# Use a lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Reduced patience
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,  # Reduced patience
    min_lr=1e-6
)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train the model with reduced data
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=10,  # Reduced epochs since we're using less data
    validation_data=validation_generator,
    validation_steps=valid_steps,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Save class indices for later use
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# Evaluate the model with reduced test data
test_loss, test_accuracy = model.evaluate(
    test_generator,
    steps=test_steps
)
print(f"\nTest accuracy (on {SUBSET_FRACTION*100}% of test data): {test_accuracy:.4f}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Save the final model explicitly
model_save_path = os.path.join(os.path.dirname(__file__), 'best_model.h5')
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# Also save a backup copy
backup_model_path = os.path.join(os.path.dirname(__file__), 'model_backup.h5')
model.save(backup_model_path)
print(f"Backup model saved to: {backup_model_path}")

