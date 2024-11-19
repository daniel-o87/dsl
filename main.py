import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import pydicom
import math
import zipfile
import subprocess
from datetime import datetime
import gc


def setup_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Found {len(gpus)} physical gpus, {len(logical_gpus)} logical gpus")
        except RuntimeError as e:
            print(e)
    else:
        print("no gpus")
    return gpus

def verify_data_paths(base_dir, train_dir, test_dir):
    print(f"Base dir {os.path.abspath(base_dir)}")
    print(f"Train dir {os.path.abspath(train_dir)}")
    print(f"Test dir {os.path.abspath(test_dir)}")

    if not os.path.exists(train_dir):
        raise ValueError(f"Train dir not found {train_dir}")

    if not os.path.exists(test_dir):
        raise ValueError(f"Test dir not found {test_dir}")

def analyze_class_distribution(df):
    """Analyze and print class distribution information"""
    distribution = df['target'].value_counts(normalize=True)
    print("\nClass Distribution:")
    print(distribution)
    print("\nTotal samples:", len(df))
    print("Positive samples:", len(df[df['target'] == 1]))
    print("Negative samples:", len(df[df['target'] == 0]))
    return distribution

# DICOM to Array Conversion
def dicom_to_array(file_path):
    try:
        dicom = pydicom.dcmread(file_path)
        img = dicom.pixel_array
        # Normalize to 0-255 range
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        # Handle different image formats
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            img = img[..., :3]
        return img
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def prepare_metadata(dataframe, is_train=True):
    dataframe = dataframe.copy()

    anatom_site_col = "anatom_site_general_challenge"

    dataframe['sex'] = dataframe['sex'].fillna('unknown')
    dataframe[anatom_site_col] = dataframe[anatom_site_col].fillna('unknown')
    dataframe['age_approx'] = dataframe['age_approx'].fillna(dataframe['age_approx'].mean())

    # Encode categorical variables
    categorical_cols = ['sex', anatom_site_col]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(dataframe[categorical_cols])

    # Scale numerical variables
    numerical_cols = ['age_approx']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dataframe[numerical_cols])

    # Combine encoded categorical and scaled numerical features
    metadata_features = np.hstack([encoded_features, scaled_features])

    return metadata_features

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, images_dir, metadata, batch_size=16, shuffle=True, **kwargs):
        super().__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_filenames = self.dataframe['filename'].values
        self.labels = self.dataframe['target'].values.astype(np.float32)
        self.images_dir = os.path.abspath(images_dir)
        self.metadata = metadata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))

        if not os.path.exists(self.images_dir):
            raise ValueError(f"Img dir not found {self.images_dir}")

        print(f"Image directory: {self.images_dir}")
        print(f"Num of images: {len(self.image_filenames)}")

        if len(self.image_filenames) > 0:
            first_file = os.path.join(self.images_dir, self.image_filenames[0])
            print(f"First file path: {first_file}\nFirst file exists: {os.path.exists(first_file)}")

        self.valid_indexes = self._validate_files()
        self.on_epoch_end()

    def _validate_files(self):
        """Validate which files exist and return their indexes"""
        valid_indexes = []
        for idx, fname in enumerate(self.image_filenames):
            file_path = os.path.join(self.images_dir, fname)
            if os.path.exists(file_path):
                valid_indexes.append(idx)
            else:
                print(f"Skipping missing file: {file_path}")
        return np.array(valid_indexes)

    def __len__(self):
        return math.ceil(len(self.valid_indexes) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.valid_indexes))
        batch_indexes = self.valid_indexes[start_idx:end_idx]
        
        batch_images = []
        valid_indexes = []
        valid_labels = []
        valid_metadata = []

        for i, index in enumerate(batch_indexes):
            file_path = os.path.join(self.images_dir, self.image_filenames[index])
            img_array = dicom_to_array(file_path)
            
            if img_array is not None:
                batch_images.append(img_array / 255.0)
                valid_indexes.append(i)
                valid_labels.append(self.labels[index])
                valid_metadata.append(self.metadata[index])

        if not batch_images:  # If no valid images in batch
            # Return a minimal valid batch to prevent training failure
            return {'image_input': np.zeros((1, 224, 224, 3)), 
                   'metadata_input': self.metadata[0:1]}, np.array([0.0])

        return {
            'image_input': np.array(batch_images),
            'metadata_input': np.array(valid_metadata)
        }, np.array(valid_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_indexes)

def create_model(metadata_shape):
    # Image input
    image_input = layers.Input(shape=(224, 224, 3), name='image_input')
    x = layers.Conv2D(32, 3, activation='relu')(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Flatten()(x)

    # Metadata input
    metadata_input = layers.Input(shape=(metadata_shape,), name='metadata_input')
    y = layers.Dense(64, activation='relu')(metadata_input)

    # Combine image and metadata
    combined = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(1, activation='sigmoid')(z)

    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

def compute_class_weights(y):
    """Compute balanced class weights"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(class_weights))

# Custom callback for model saving
class ModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        # Save model after each epoch
        epoch_model_path = os.path.join(self.output_dir, f'model_epoch_{self.epoch_count}.h5')
        try:
            self.model.save(epoch_model_path)
            print(f"\nSaved model for epoch {self.epoch_count} to: {epoch_model_path}")
            # Verify the save
            if os.path.exists(epoch_model_path):
                size = os.path.getsize(epoch_model_path)
                print(f"Verified save: {size} bytes")
            else:
                print("Warning: Model file not found after save!")
        except Exception as e:
            print(f"Error saving model for epoch {self.epoch_count}: {str(e)}")
            # Try saving weights at least
            try:
                weights_path = os.path.join(self.output_dir, f'weights_epoch_{self.epoch_count}.h5')
                self.model.save_weights(weights_path)
                print(f"Saved weights as fallback to: {weights_path}")
            except Exception as e:
                print(f"Error saving weights: {str(e)}")

def train_model(train_generator, val_generator, class_weights, epochs=3):
    try:
        metadata_shape = train_generator.metadata.shape[1]
        model = create_model(metadata_shape)

        # Create a timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"training_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCreated output directory: {output_dir}")

        # Initialize callbacks
        model_saver = ModelSaver(output_dir)
        
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv')
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )

        # Save initial model
        print("\nSaving initial model...")
        initial_model_path = os.path.join(output_dir, 'initial_model.h5')
        model.save(initial_model_path)
        print(f"Initial model saved to: {initial_model_path}")

        # Train model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=[model_saver, early_stopping, csv_logger],
            verbose=1
        )

        # Save final model in multiple formats
        print("\nSaving final models...")
        
        # Save in H5 format
        h5_path = os.path.join(output_dir, 'final_model.h5')
        model.save(h5_path)
        print(f"Saved H5 model to: {h5_path}")

        # Save in Keras format
        keras_path = os.path.join(output_dir, 'final_model.keras')
        model.save(keras_path)
        print(f"Saved Keras model to: {keras_path}")

        # Save weights separately
        weights_path = os.path.join(output_dir, 'final_weights.h5')
        model.save_weights(weights_path)
        print(f"Saved weights to: {weights_path}")

        # Verify all saves
        files_to_check = [
            'initial_model.h5',
            'final_model.h5',
            'final_model.keras',
            'final_weights.h5'
        ]
        
        print("\nVerifying saved files:")
        for file in files_to_check:
            path = os.path.join(output_dir, file)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"Found {file}: {size} bytes")
            else:
                print(f"Missing {file}!")

        # Plot and save metrics
        plot_metrics(history, output_dir)

        return model, history

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def plot_metrics(history, output_dir):
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(history.history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    try:
        # Check write permissions
        try:
            test_file = "test_write_permission"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("Write permission verified")
        except Exception as e:
            print(f"Warning: Write permission test failed: {str(e)}")

        # Check available disk space
        import shutil
        total, used, free = shutil.disk_usage(".")
        print(f"\nDisk space available: {free // (2**30)} GB")
        if free // (2**30) < 10:  # Less than 10GB free
            print("Warning: Low disk space!")

        gpus = setup_gpus()
        BASE_DIR = os.path.expanduser(".")
        TRAIN_DIR = os.path.join(BASE_DIR, 'train')
        TEST_DIR = os.path.join(BASE_DIR, 'test')
        verify_data_paths(BASE_DIR, TRAIN_DIR, TEST_DIR)

        # Load and prepare data
        train_csv = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
        test_csv = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

        train_csv['filename'] = train_csv['image_name'].apply(lambda x: f"{x}.dcm")
        test_csv['filename'] = test_csv['image'].apply(lambda x: f"{x}.dcm")
        train_csv['target'] = train_csv['target'].astype(np.float32)

        # Analyze class distribution
        print("\nAnalyzing class distribution in training data:")
        analyze_class_distribution(train_csv)

        # Split data
        train_df, val_df = train_test_split(train_csv, test_size=0.2, random_state=42, stratify=train_csv['target'])
        
        print("\nTraining set distribution:")
        analyze_class_distribution(train_df)
        print("\nValidation set distribution:")
        analyze_class_distribution(val_df)

        # Compute class weights
        class_weights = compute_class_weights(train_df['target'])
        print("\nClass weights:", class_weights)

        # Prepare Metadata for Generators
        train_metadata = prepare_metadata(train_df, is_train=True)
        val_metadata = prepare_metadata(val_df, is_train=True)

        # Create Data Generators with Reduced Batch Size
        train_generator = DataGenerator(train_df, 'train', train_metadata, batch_size=16, shuffle=True)
        val_generator = DataGenerator(val_df, 'test', val_metadata, batch_size=16, shuffle=False)

        # Train Model
        model, history = train_model(train_generator, val_generator, class_weights, epochs=5)

        # Force cleanup
        del model
        gc.collect()
        tf.keras.backend.clear_session()

        # Verify final output
        print("\nFinal directory contents:")
        latest_output = max([d for d in os.listdir('.') if d.startswith('training_output_')], key=os.path.getmtime)
        os.system(f'ls -lh {latest_output}')

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up in case of error
        try:
            tf.keras.backend.clear_session()
        except:
            pass
        
        # Exit with error status
        exit(1)
