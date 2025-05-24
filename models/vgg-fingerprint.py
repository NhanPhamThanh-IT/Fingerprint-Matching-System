# %%
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = (160, 160)
BATCH_SIZE = 10
EPOCHS = 100
NUM_CLASSES = 10
DATA_DIR = "/kaggle/input/fingerprint-dataset-for-fvc2000-db4-b/dataset_FVC2000_DB4_B/dataset"
TRAIN_TEST_SPLIT = 0.8
FINE_TUNE_LAYERS = 3  
MODEL_DIR = "./saved_models"  

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    class_dirs = ['real_data', 'train_data']

    for class_dir in class_dirs:
        dir_path = os.path.join(data_dir, class_dir)
        for img_name in os.listdir(dir_path):
            if img_name.endswith('.bmp'):
                img_path = os.path.join(dir_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    img = img / 255.0  
                    images.append(img)
                    if '_' in img_name:
                        label = int(img_name.split('_')[0])
                    else:
                        label = int(img_name.split('.')[0])
                    labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    print("Label distribution:", np.unique(labels, return_counts=True))

    train_data, val_data, train_labels, val_labels = train_test_split(
        images, labels, test_size=1-TRAIN_TEST_SPLIT, stratify=labels, random_state=42
    )

    print("Training label distribution:", np.unique(train_labels, return_counts=True))
    print("Validation label distribution:", np.unique(val_labels, return_counts=True))

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    return train_data, train_labels, val_data, val_labels, class_weight_dict

def apply_inversion(image):
    h, w = image.shape[:2]
    part1 = image[:, :w//2]  
    part2 = image[:, w//2:] 
    part1_inv = 1.0 - part1
    part2_inv = 1.0 - part2

    part1_inv_resized = cv2.resize(part1_inv, IMG_SIZE)
    part2_inv_resized = cv2.resize(part2_inv, IMG_SIZE)
    return part1_inv_resized, part2_inv_resized

def create_data_generator():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def build_model(base_model, model_name, input_shape=(160, 160, 3), fine_tune=False):
    if fine_tune:
        for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False
        for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
            layer.trainable = True
    else:
        base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs, outputs, name=model_name)
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate(model, train_data, train_labels, val_data, val_labels, data_generator=None, class_weight=None):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    if data_generator:
        train_gen = data_generator.flow(train_data, train_labels, batch_size=BATCH_SIZE)
        history = model.fit(train_gen, epochs=EPOCHS, validation_data=(val_data, val_labels),
                           callbacks=[early_stopping], class_weight=class_weight)
    else:
        history = model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS,
                           validation_data=(val_data, val_labels), callbacks=[early_stopping], class_weight=class_weight)
    
    val_pred = model.predict(val_data)
    val_pred_classes = np.argmax(val_pred, axis=1)
    print(f"Classification Report for {model.name}:\n")
    print(classification_report(val_labels, val_pred_classes))
    
    cm = confusion_matrix(val_labels, val_pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model.name}')
    plt.colorbar()
    plt.show()
    
    model_path = os.path.join(MODEL_DIR, f"{model.name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return history

def main():
    train_data, train_labels, val_data, val_labels, class_weight_dict = load_and_preprocess_data(DATA_DIR)
    data_generator = create_data_generator()
    
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    
    vgg16_normal = build_model(vgg16_base, 'VGG16_Normal')
    vgg19_normal = build_model(vgg19_base, 'VGG19_Normal')
    vgg16_inversion = build_model(vgg16_base, 'VGG16_Inversion', input_shape=(160, 160, 3), fine_tune=True)
    vgg19_inversion = build_model(vgg19_base, 'VGG19_Inversion', input_shape=(160, 160, 3), fine_tune=True)
    vgg16_multi_aug = build_model(vgg16_base, 'VGG16_Multi_Augmentation', fine_tune=True)
    vgg19_multi_aug = build_model(vgg19_base, 'VGG19_Multi_Augmentation', fine_tune=True)
    
    print("Training VGG16 Normal...")
    history_vgg16_normal = train_and_evaluate(vgg16_normal, train_data, train_labels, val_data, val_labels, class_weight=class_weight_dict)
    print("Training VGG19 Normal...")
    history_vgg19_normal = train_and_evaluate(vgg19_normal, train_data, train_labels, val_data, val_labels, class_weight=class_weight_dict)
    
    print("Training VGG16 with Inversion...")
    train_data_inv = []
    train_labels_inv = []
    for img, lbl in zip(train_data, train_labels):
        part1, part2 = apply_inversion(img)
        train_data_inv.extend([part1, part2])
        train_labels_inv.extend([lbl, lbl])
    train_data_inv = np.array(train_data_inv)
    train_labels_inv = np.array(train_labels_inv)
    val_data_inv = np.array([cv2.resize(apply_inversion(img)[0], IMG_SIZE) for img in val_data])
    history_vgg16_inv = train_and_evaluate(vgg16_inversion, train_data_inv, train_labels_inv, val_data_inv, val_labels, class_weight=class_weight_dict)
    
    print("Training VGG19 with Inversion...")
    history_vgg19_inv = train_and_evaluate(vgg19_inversion, train_data_inv, train_labels_inv, val_data_inv, val_labels, class_weight=class_weight_dict)
    
    print("Training VGG16 with Multi-Augmentation...")
    history_vgg16_aug = train_and_evaluate(vgg16_multi_aug, train_data, train_labels, val_data, val_labels, data_generator, class_weight=class_weight_dict)
    print("Training VGG19 with Multi-Augmentation...")
    history_vgg19_aug = train_and_evaluate(vgg19_multi_aug, train_data, train_labels, val_data, val_labels, data_generator, class_weight=class_weight_dict)
    
    def plot_history(histories, title):
        plt.figure(figsize=(12, 4))
        for name, history in histories.items():
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label=f'{name} Train')
            plt.plot(history.history['val_accuracy'], label=f'{name} Val')
            plt.title('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label=f'{name} Train')
            plt.plot(history.history['val_loss'], label=f'{name} Val')
            plt.title('Loss')
            plt.legend()
        plt.suptitle(title)
        plt.show()
    
    plot_history({
        'VGG16 Normal': history_vgg16_normal,
        'VGG19 Normal': history_vgg19_normal,
        'VGG16 Inversion': history_vgg16_inv,
        'VGG19 Inversion': history_vgg19_inv,
        'VGG16 Multi-Aug': history_vgg16_aug,
        'VGG19 Multi-Aug': history_vgg19_aug
    }, 'Model Performance Comparison')

if __name__ == "__main__":
    main()


