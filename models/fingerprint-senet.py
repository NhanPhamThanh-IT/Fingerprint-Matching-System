# %%
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Multiply, Reshape, Layer
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPUs detected:", physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPUs detected. Running on CPU.")

set_global_policy('mixed_float16')

class SENetLayer(Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SENetLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.squeeze = GlobalAveragePooling2D()
        self.excitation = tf.keras.Sequential([
            Dense(channel // self.ratio, activation='relu'),
            Dense(channel, activation='sigmoid')
        ])
        super(SENetLayer, self).build(input_shape)

    def call(self, inputs):
        squeezed = self.squeeze(inputs)
        squeezed = tf.expand_dims(tf.expand_dims(squeezed, 1), 1)
        
        excitation = self.excitation(squeezed)
        
        scaled = Multiply()([inputs, excitation])
        return scaled

def efficientnet_b0_plus(input_shape=(224, 224, 3), num_classes=10):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.get_layer('block7a_project_bn').output
    x = SENetLayer()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='swish')(x)
    output = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

def mobilenet_plus(input_shape=(224, 224, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    def inverted_residual_block(x, filters, stride, expansion):
        in_channels = x.shape[-1]
        x_expanded = Conv2D(expansion * in_channels, (1, 1), padding='same')(x)
        x_expanded = layers.BatchNormalization()(x_expanded)
        x_expanded = layers.Activation('relu')(x_expanded)
        
        x_depthwise = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same')(x_expanded)
        x_depthwise = layers.BatchNormalization()(x_depthwise)
        x_depthwise = layers.Activation('relu')(x_depthwise)
        
        x_pointwise = Conv2D(filters, (1, 1), padding='same')(x_depthwise)
        x_pointwise = layers.BatchNormalization()(x_pointwise)
        
        x = SENetLayer()(x_pointwise)
        
        if stride == 1 and in_channels == filters:
            x = layers.Add()([x, x_pointwise])
        return x
    
    x = inverted_residual_block(x, 64, 1, 6)
    x = inverted_residual_block(x, 128, 2, 6)
    x = inverted_residual_block(x, 128, 1, 6)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model

def load_and_preprocess_data(data_dir):
    img_train = np.load(os.path.join(data_dir, 'np_data', 'img_train.npy'))
    label_train = np.load(os.path.join(data_dir, 'np_data', 'label_train.npy'))
    img_real = np.load(os.path.join(data_dir, 'np_data', 'img_real.npy'))
    label_real = np.load(os.path.join(data_dir, 'np_data', 'label_real.npy'))
    
    images = np.concatenate((img_train, img_real), axis=0)
    labels = np.concatenate((label_train, label_real), axis=0)
    
    processed_images = []
    for img in images:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = tf.image.resize(img, [224, 224]).numpy()
        processed_images.append(img)
    
    return np.array(processed_images), labels

if __name__ == "__main__":
    data_dir = '/kaggle/input/fingerprint-dataset-for-fvc2000-db4-b/dataset_FVC2000_DB4_B/dataset'
    images, labels = load_and_preprocess_data(data_dir)
    
    images = images / 255.0
    num_classes = len(np.unique(labels))
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    effnet_model = efficientnet_b0_plus(num_classes=num_classes)
    mobilenet_model = mobilenet_plus(num_classes=num_classes)
    
    effnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mobilenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    batch_size = 64
    
    effnet_model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
                     epochs=100, 
                     validation_data=(X_test, y_test))
    
    mobilenet_model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
                        epochs=100, 
                        validation_data=(X_test, y_test))
    
    effnet_loss, effnet_acc = effnet_model.evaluate(X_test, y_test)
    mobilenet_loss, mobilenet_acc = mobilenet_model.evaluate(X_test, y_test)
    
    print(f"EfficientNet-B0+ Accuracy: {effnet_acc:.4f}")
    print(f"MobileNet+ Accuracy: {mobilenet_acc:.4f}")

    effnet_model.save('efficientnet_b0_plus.h5')

    mobilenet_model.save('mobilenet_plus.h5')

    print("Đã lưu các mô hình thành công!")


