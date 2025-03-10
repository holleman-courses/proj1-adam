import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('projtest_model.h5')

# Load the dataset (same as training data)
eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_directory(
    'HW4Data/train',  # Same directory as training (tiny dataset)
    target_size=(150, 150),
    batch_size=2,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(eval_generator)
print(f"Evaluation Loss: {loss:.4f}")
print(f"Evaluation Accuracy: {accuracy:.4f}")