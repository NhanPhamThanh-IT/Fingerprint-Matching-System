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

class DualAttentionLayer(Layer):
    def __init__(self, channel_reduction_factor=16, **kwargs):
        super(DualAttentionLayer, self).__init__(**kwargs)
        self.channel_reduction_factor = channel_reduction_factor

    def build(self, input_shape):
        channel = input_shape[-1]
        self.query_conv = Conv2D(channel // self.channel_reduction_factor, kernel_size=1, padding='same')
        self.key_conv = Conv2D(channel // self.channel_reduction_factor, kernel_size=1, padding='same')
        self.value_conv = Conv2D(channel, kernel_size=1, padding='same')
        self.channel_query_conv = Conv2D(channel // self.channel_reduction_factor, kernel_size=1, padding='same')
        self.channel_key_conv = Conv2D(channel // self.channel_reduction_factor, kernel_size=1, padding='same')
        self.channel_value_conv = Conv2D(channel // self.channel_reduction_factor, kernel_size=1, padding='same')
        self.channel_projection = Conv2D(channel, kernel_size=1, padding='same')
        super(DualAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size, height, width, channel = tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]

        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        query = tf.reshape(query, [batch_size, height * width, channel // self.channel_reduction_factor])
        key = tf.reshape(key, [batch_size, height * width, channel // self.channel_reduction_factor])
        value = tf.reshape(value, [batch_size, height * width, channel])

        pos_attention_scores = tf.matmul(query, key, transpose_b=True)
        scaling_factor = tf.sqrt(tf.cast(channel // self.channel_reduction_factor, inputs.dtype))
        pos_attention_scores = tf.nn.softmax(pos_attention_scores / scaling_factor)
        pos_out = tf.matmul(pos_attention_scores, value)
        pos_out = tf.reshape(pos_out, [batch_size, height, width, channel])

        channel_query = self.channel_query_conv(inputs)
        channel_key = self.channel_key_conv(inputs)
        channel_value = self.channel_value_conv(inputs)

        channel_query = tf.reshape(channel_query, [batch_size, height * width, channel // self.channel_reduction_factor])
        channel_key = tf.reshape(channel_key, [batch_size, height * width, channel // self.channel_reduction_factor])
        channel_value = tf.reshape(channel_value, [batch_size, height * width, channel // self.channel_reduction_factor])

        channel_query = tf.transpose(channel_query, perm=[0, 2, 1])
        channel_key = tf.transpose(channel_key, perm=[0, 2, 1])
        channel_value = tf.transpose(channel_value, perm=[0, 2, 1])

        chan_attention_scores = tf.matmul(channel_key, channel_query, transpose_b=True)
        chan_attention_scores = tf.nn.softmax(chan_attention_scores / scaling_factor)
        chan_out = tf.matmul(chan_attention_scores, channel_value)
        chan_out = tf.transpose(chan_out, perm=[0, 2, 1])
        chan_out = tf.reshape(chan_out, [batch_size, height, width, channel // self.channel_reduction_factor])

        chan_out = self.channel_projection(chan_out)

        out = tf.add(pos_out, chan_out)
        out = tf.add(inputs, out)
        return out

def efficientnet_b0_plus(input_shape=(224, 224, 3), num_classes=10):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.get_layer('block7a_project_bn').output
    x = DualAttentionLayer(channel_reduction_factor=16)(x)
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
        
        if filters == 128 and stride == 1:
            x = DualAttentionLayer(channel_reduction_factor=16)(x_pointwise)
        else:
            x = x_pointwise
        
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
    
    tf.keras.backend.clear_session()
    
    effnet_model = efficientnet_b0_plus(num_classes=num_classes)
    mobilenet_model = mobilenet_plus(num_classes=num_classes)
    
    effnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mobilenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    batch_size = 8
    
    effnet_model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
                     epochs=100, 
                     validation_data=(X_test, y_test))
    
    effnet_model.save('effnet_model.h5')
    
    tf.keras.backend.clear_session()
    
    mobilenet_model = mobilenet_plus(num_classes=num_classes)
    mobilenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    mobilenet_model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
                        epochs=100, 
                        validation_data=(X_test, y_test))
    
    mobilenet_model.save('mobilenet_model.h5')
    
    effnet_loss, effnet_acc = effnet_model.evaluate(X_test, y_test)
    mobilenet_loss, mobilenet_acc = mobilenet_model.evaluate(X_test, y_test)
    
    print(f"EfficientNet-B0+ Accuracy: {effnet_acc:.4f}")
    print(f"MobileNet+ Accuracy: {mobilenet_acc:.4f}")


