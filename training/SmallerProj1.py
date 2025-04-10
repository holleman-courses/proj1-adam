import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

class Config:
    BASE_DIR = 'Cars_Dataset'
    IMAGE_SIZE = 96
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    NUM_CLASSES = 2
    EPOCHS = 10
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = 'custom_cnn_gray_model.keras'
    TFLITE_MODEL_PATH = 'custom_cnn_gray_model.tflite'
    QUANTIZED_TFLITE_MODEL_PATH = 'custom_cnn_gray_model_quantized.tflite'

def build_custom_cnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D()(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

def prepare_data_generators(config):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=config.VALIDATION_SPLIT
    )

    train_generator = datagen.flow_from_directory(
        config.BASE_DIR,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        config.BASE_DIR,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    return train_generator, val_generator, dict(enumerate(class_weights))

def train_model(config, model, train_gen, val_gen, class_weights):
    checkpoint = ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.2, min_lr=1e-6)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    return model, history

def evaluate_model(model, val_generator):
    results = model.evaluate(val_generator)
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    print("\nClassification Report:\n", classification_report(
        y_true, y_pred, target_names=list(val_generator.class_indices.keys())
    ))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(val_generator.class_indices.keys()),
                yticklabels=list(val_generator.class_indices.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('custom_cnn_confusion_matrix.png')
    plt.close()

    return {
        'loss': results[0],
        'accuracy': results[1]
    }

def convert_to_tflite(model, config, quantize=False):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model (tf.keras.Model): Trained Keras model
        config (Config): Configuration object
        quantize (bool): Whether to apply quantization
    
    Returns:
        bytes: TFLite model
    """
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Specify representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                # Generate random input data matching the model's input shape (grayscale)
                data = np.random.rand(1, config.IMAGE_SIZE, config.IMAGE_SIZE, 1).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        
        # Restrict to integer-only quantization for supported operations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]


    converter.allow_custom_ops = True
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    save_path = config.QUANTIZED_TFLITE_MODEL_PATH if quantize else config.TFLITE_MODEL_PATH
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite {'Quantized' if quantize else ''} Model saved to {save_path}")
    
    return tflite_model

def validate_tflite_model(config, tflite_model, val_generator, is_quantized=False):
    """
    Validate TFLite model performance
    
    Args:
        config (Config): Configuration object
        tflite_model (bytes): TFLite model
        val_generator (tf.keras.preprocessing.image.DirectoryIterator): Validation data generator
        is_quantized (bool): Whether the model is quantized
    
    Returns:
        dict: Model performance metrics
    """
    # Prepare interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare validation data
    val_generator.reset()
    predictions = []
    true_labels = val_generator.classes
    
    # Iterate through validation data
    for i in range(len(val_generator)):
        batch_images, _ = val_generator[i]
        
        for image in batch_images:
            # Preprocess image (for grayscale, make sure it has a single channel)
            input_data = np.expand_dims(image, axis=0)
            
            if is_quantized:
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = ((input_data / input_scale) + input_zero_point).astype(np.uint8)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])

            if is_quantized:
                output_scale, output_zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

            predictions.append(np.argmax(output_data))
    
    # Compute metrics
    report = classification_report(
        true_labels[:len(predictions)], 
        predictions, 
        target_names=list(val_generator.class_indices.keys())
    )
    print("\nTFLite Model Classification Report:\n", report)
    
    return {
        'predictions': predictions,
        'true_labels': true_labels[:len(predictions)]
    }

def main():
    # Configuration
    config = Config()
    
    # Determine input shape for grayscale (1 channel instead of 3)
    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
    
    # Create Model
    model = build_custom_cnn(input_shape, config.NUM_CLASSES)
    model.summary()
    
    # Prepare Data
    train_generator, val_generator, class_weights = prepare_data_generators(config)
    print("Class Indices:", train_generator.class_indices)
    
    # Train Model
    trained_model, history = train_model(
        config, 
        model, 
        train_generator, 
        val_generator, 
        class_weights
    )
    
    # Evaluate Model
    metrics = evaluate_model(trained_model, val_generator)
    print("\nModel Metrics:", metrics)
    
    # Convert to TFLite (non-quantized)
    tflite_model = convert_to_tflite(trained_model, config, quantize=False)
    
    # Convert to Quantized TFLite
    quantized_tflite_model = convert_to_tflite(trained_model, config, quantize=True)
    
    # Validate TFLite Models
    print("\nValidating Non-Quantized TFLite Model:")
    non_quantized_results = validate_tflite_model(config, tflite_model, val_generator, is_quantized=False)
    
    print("\nValidating Quantized TFLite Model:")
    quantized_results = validate_tflite_model(config, quantized_tflite_model, val_generator, is_quantized=True)
    
    print("\nModel Training, Conversion, and Validation Complete!")

if __name__ == "__main__":
    main()
