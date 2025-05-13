import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Multiply, Reshape, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import custom_object_scope

# Custom CBAMLayer class
class CBAMLayer(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = Dense(channel // self.ratio, activation='relu')
        self.shared_layer_two = Dense(channel)
        self.spatial_attention_conv = Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        super(CBAMLayer, self).build(input_shape)

    def call(self, inputs):
        channel = inputs.shape[-1]
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        channel_attention = tf.keras.activations.sigmoid(avg_pool + max_pool)
        channel_refined = Multiply()([inputs, channel_attention])

        avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        concat_spatial = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_attention = self.spatial_attention_conv(concat_spatial)
        spatial_refined = Multiply()([channel_refined, spatial_attention])

        return spatial_refined

    def get_config(self):
        config = super(CBAMLayer, self).get_config()
        config.update({'ratio': self.ratio})
        return config

# Custom SENetLayer class
class SENetLayer(tf.keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SENetLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_pool = GlobalAveragePooling2D()
        self.fc1 = Dense(channels // self.ratio, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')
        super(SENetLayer, self).build(input_shape)

    def call(self, inputs):
        x = self.global_pool(inputs)
        x = Reshape((1, 1, -1))(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return Multiply()([inputs, x])

    def get_config(self):
        config = super(SENetLayer, self).get_config()
        config.update({'ratio': self.ratio})
        return config

# Custom DualAttentionLayer class
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

    def get_config(self):
        config = super(DualAttentionLayer, self).get_config()
        config.update({'channel_reduction_factor': self.channel_reduction_factor})
        return config

# Custom SelfAttentionLayer class
class SelfAttentionLayer(Layer):
    def __init__(self, channel_reduction_factor=16, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.channel_reduction_factor = channel_reduction_factor

    def build(self, input_shape):
        channel = input_shape[-1]
        self.query_conv = Conv2D(channel // self.channel_reduction_factor, kernel_size=1, padding='same')
        self.key_conv = Conv2D(channel // self.channel_reduction_factor, kernel_size=1, padding='same')
        self.value_conv = Conv2D(channel, kernel_size=1, padding='same')
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size, height, width, channel = tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)
        query = tf.reshape(query, [batch_size, height * width, channel // self.channel_reduction_factor])
        key = tf.reshape(key, [batch_size, height * width, channel // self.channel_reduction_factor])
        value = tf.reshape(value, [batch_size, height * width, channel])
        attention_scores = tf.matmul(query, key, transpose_b=True)
        scaling_factor = tf.sqrt(tf.cast(channel // self.channel_reduction_factor, inputs.dtype))
        attention_scores = tf.nn.softmax(attention_scores / scaling_factor)
        attention_out = tf.matmul(attention_scores, value)
        attention_out = tf.reshape(attention_out, [batch_size, height, width, channel])
        out = tf.add(inputs, attention_out)
        return out

    def get_config(self):
        config = super(SelfAttentionLayer, self).get_config()
        config.update({'channel_reduction_factor': self.channel_reduction_factor})
        return config

# MODEL_PATHS dictionary
MODEL_PATHS = {
    "MobileNet+SENet": {"path": r"mobilenet_plus+Snet.h5", "input_size": (224, 224)},
    "EfficientNetB0+SENet": {"path": r"efficientnet_b0_plus+Snet.h5", "input_size": (224, 224)},
    "MobileNet+SelfAttention": {"path": r"mobilenet_plus+Self-Attention.h5", "input_size": (224, 224)},
    "EfficientNetB0+SelfAttention": {"path": r"efficientnet_b0_plus+Self-Attention.h5", "input_size": (224, 224)},
    "MobileNet+DualAttention": {"path": r"mobilenet_plus+dual.h5", "input_size": (224, 224)},
    "EfficientNetB0+DualAttention": {"path": r"mobilenet_plus+dual.h5", "input_size": (224, 224)},
    "MobileNet+CBAM": {"path": r"mobilenet_plus+CBAM.keras", "input_size": (224, 224)},
    "EfficientNetB0+CBAM": {"path": r"efficientnet_b0_plus+CBAM.keras", "input_size": (224, 224)},
    "VGG16+Inversion": {"path": r"VGG16_Inversion.h5", "input_size": (160, 160)},
    "VGG16+MultiAugmentation": {"path": r"VGG16_Multi_Augmentation.h5", "input_size": (160, 160)},
    "VGG16+Normal": {"path": r"VGG16_Normal.h5", "input_size": (160, 160)},
    "VGG19+Inversion": {"path": r"VGG19_Inversion.h5", "input_size": (160, 160)},
    "VGG19+MultiAugmentation": {"path": r"VGG19_Multi_Augmentation.h5", "input_size": (160, 160)},
    "VGG19+Normal": {"path": r"VGG19_Normal.h5", "input_size": (160, 160)},
}

def preprocess_image(image, input_size=(224, 224)):
    try:
        img = np.array(image)
        if len(img.shape) == 2 or img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[-1] != 3:
            raise ValueError(f"Unexpected number of channels: {img.shape[-1]}")
        img = cv2.resize(img, input_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def load_model(model_path, model_name):
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        file_size = os.path.getsize(model_path)
        if file_size < 1024:
            st.error(f"Model file is too small or corrupted: {model_path} (Size: {file_size} bytes)")
            return None

        # Placeholder for 'Cast' layer
        class CastLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(CastLayer, self).__init__(**kwargs)
            def call(self, inputs):
                return tf.cast(inputs, tf.float32)
            def get_config(self):
                return super(CastLayer, self).get_config()

        # Load model with custom objects
        with custom_object_scope({
            'CBAMLayer': CBAMLayer,
            'SENetLayer': SENetLayer,
            'DualAttentionLayer': DualAttentionLayer,
            'SelfAttentionLayer': SelfAttentionLayer,
            'Cast': CastLayer
        }):
            model = tf.keras.models.load_model(model_path)
            # Try to extract features from an earlier layer
            feature_layer_name = None
            for layer in model.layers[::-1]:
                if 'global_average_pooling2d' in layer.name or 'dense' in layer.name:
                    feature_layer_name = layer.name
                    break
            if feature_layer_name:
                model = Model(inputs=model.input, outputs=model.get_layer(feature_layer_name).output)
            else:
                st.warning("No suitable feature extraction layer found. Using model output directly.")

        st.success(f"Model loaded successfully: {model_name}")
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)} (Path: {model_path})")
        return None

def identify_fingerprint(input_img, dataset_images, model, input_size, threshold=0.8):
    try:
        input_processed = preprocess_image(input_img, input_size)
        if input_processed is None:
            return None, None, None
        input_features = model.predict(input_processed)

        similarities = []
        for dataset_img, filename in dataset_images:
            person_name = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
            dataset_processed = preprocess_image(dataset_img, input_size)
            if dataset_processed is None:
                st.warning(f"Skipping dataset image {filename} due to preprocessing error")
                continue
            dataset_features = model.predict(dataset_processed)

            similarity = np.dot(input_features[0], dataset_features[0]) / (
                np.linalg.norm(input_features[0]) * np.linalg.norm(dataset_features[0])
            )
            similarities.append((person_name, similarity, filename))

        if not similarities:
            return None, None, None

        # Find the highest similarity score
        max_similarity = max(similarity for _, similarity, _ in similarities)
        
        # Collect all matches with the highest similarity score
        best_matches = [
            (person, similarity, filename)
            for person, similarity, filename in similarities
            if similarity == max_similarity
        ]

        # Check if the highest similarity is above the threshold
        if max_similarity >= threshold:
            return best_matches, max_similarity, [filename for _, _, filename in best_matches]
        else:
            return None, max_similarity, None

    except Exception as e:
        st.error(f"Error during fingerprint identification: {str(e)}")
        return None, None, None

# Streamlit app
st.title("Fingerprint Identification Application")

# Model selection
model_name = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
model_info = MODEL_PATHS[model_name]
model_path = model_info["path"]
input_size = model_info["input_size"]

st.write(f"Using model: {model_name}")
st.write(f"Expected input size: {input_size}")
st.write("Similarity Threshold: 0.8")

# File uploaders
input_img_file = st.file_uploader("Upload Fingerprint Image to Identify", type=['bmp', 'png', 'jpg', 'jpeg'], key="input_img")
dataset_img_files = st.file_uploader("Upload Dataset Fingerprint Images", type=['bmp', 'png', 'jpg', 'jpeg'], accept_multiple_files=True, key="dataset_imgs")

# Clear dataset button
if st.button("Clear Dataset"):
    # Safely clear dataset-related session state keys
    if "dataset_imgs" in st.session_state:
        del st.session_state["dataset_imgs"]
    if "dataset_files" in st.session_state:
        del st.session_state["dataset_files"]
    st.rerun()

if input_img_file and dataset_img_files:
    input_img = Image.open(input_img_file)
    st.image(input_img, caption="Input Fingerprint", width=300)

    dataset_images = [(Image.open(dataset_img_file), dataset_img_file.name) for dataset_img_file in dataset_img_files]    
    # Display Uploaded Dataset Images with Grid and Pagination
    if dataset_images:
        st.subheader("Uploaded Dataset Images")
        images_per_page = 10
        total_images = len(dataset_images)
        total_pages = (total_images + images_per_page - 1) // images_per_page
        
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=max(total_pages, 1),
            value=1,
            step=1,
            key="dataset_page"
        )
        
        start_idx = (page - 1) * images_per_page
        end_idx = min(start_idx + images_per_page, total_images)
        
        cols_per_row = 5
        for i in range(start_idx, end_idx, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < end_idx:
                    img, filename = dataset_images[i + j]
                    with col:
                        st.image(img, caption=filename, width=120)
        
        st.write(f"Showing page {page} of {total_pages}")
    else:
        st.write("No dataset images uploaded.")

    if st.button("Identify Fingerprint"):
        model = load_model(model_path, model_name)
        if model:
            with st.spinner("Identifying fingerprint..."):
                best_matches, max_similarity, matched_files = identify_fingerprint(input_img, dataset_images, model, input_size)
                if best_matches:
                    st.subheader("Identification Results")
                    st.write(f"Model: {model_name}")
                    st.write(f"Similarity Score: {max_similarity:.4f}")
                    
                    # Display all matched persons and images
                    matched_persons = list(set(person for person, _, _ in best_matches))
                    st.write("Matched Image(s):")
                    
                    # Display all matched images in a grid with similarity scores
                    cols_per_row = 5
                    cols = st.columns(min(len(matched_files), cols_per_row))
                    for idx, filename in enumerate(matched_files):
                        matched_img = next(img for img, fname in dataset_images if fname == filename)
                        # Find the similarity score for this specific filename
                        similarity_score = next(similarity for _, similarity, fname in best_matches if fname == filename)
                        with cols[idx % cols_per_row]:
                            st.image(matched_img, caption=f"{filename} (Similarity: {similarity_score:.4f})", width=120)
                    
                else:
                    st.subheader("Identification Results")
                    st.write(f"Highest Similarity Score: {max_similarity:.4f}")
                    st.write("Threshold: 0.8")
                    st.error("No match found. Similarity score is below the threshold.")

st.markdown("""
### Instructions
1. Select a model from the dropdown menu.
2. Upload a single fingerprint image to identify (.bmp, .png, .jpg, or .jpeg).
3. Upload multiple fingerprint images as the dataset (named as `person_name_imageX.ext` or `person_name.ext`).
4. Click "Clear Dataset" to remove all uploaded dataset images.
5. Click "Identify Fingerprint" to see the identification results.
6. If the highest similarity score is above the threshold (0.8), the app will show all images with that score and the corresponding person(s).
7. If below the threshold, it will indicate no match was found.
""")