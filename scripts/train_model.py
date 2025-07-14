import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

img_height, img_width = 150, 150
batch_size = 16
epochs = 13  


train_dir = os.path.join(project_root, 'data', 'train')
val_dir   = os.path.join(project_root, 'data', 'validation')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(1e-4),
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=val_gen.samples // batch_size
)


model_path    = os.path.join(project_root, 'household_classifier.h5')
accuracy_path = os.path.join(project_root, 'accuracy.png')
loss_path     = os.path.join(project_root, 'loss.png')


model.save(model_path)

acc     = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss    = history.history['loss']
val_loss= history.history['val_loss']
epochs_range = range(epochs)

plt.figure()
plt.plot(epochs_range, acc,    label='train_acc')
plt.plot(epochs_range, val_acc, label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.savefig(accuracy_path)

plt.figure()
plt.plot(epochs_range, loss,     label='train_loss')
plt.plot(epochs_range, val_loss,  label='val_loss')
plt.legend()
plt.title('Loss')
plt.savefig(loss_path)

print(f'Training complete. Model saved to:\n  {model_path}\nPlots saved to:\n  {accuracy_path}\n  {loss_path}')
