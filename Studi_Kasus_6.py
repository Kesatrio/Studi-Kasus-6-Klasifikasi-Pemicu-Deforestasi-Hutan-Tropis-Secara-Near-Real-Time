import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
def generate_synthetic_satellite_stack(num_samples=1000):
    np.random.seed(42)
    X = np.random.rand(num_samples, 32, 32, 6).astype(np.float32)
    y = np.random.randint(0, 4, size=(num_samples,))
    return X, y

X, y = generate_synthetic_satellite_stack(num_samples=1200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def build_deforestation_driver_cnn(input_shape=(32, 32, 6), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
model = build_deforestation_driver_cnn()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
print("\n--- Memulai Pelatihan Model CNN ---")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
print("\n--- Evaluasi Performa Model ---")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
target_names = ['Agriculture', 'Selective Logging', 'Mining', 'Wildfire']
print("\nClassification Report (Termasuk Macro F1-Score):")
print(classification_report(y_test, y_pred, target_names=target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
