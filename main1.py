import os
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load SaMi-Trop dataset from HDF5 and CSV files
hdf5_path = "C:\\Users\\rjhha\\OneDrive\\Desktop\\identity\\Dr. Jha\\physionet-samitrop\\exams.hdf5"
csv_path = "C:\\Users\\rjhha\\OneDrive\\Desktop\\identity\\Dr. Jha\\physionet-samitrop\\exams.hdf5"
chagas_labels_path = "C:\\Users\\rjhha\\OneDrive\\Desktop\\identity\\Dr. Jha\\physionet-samitrop\\samitrop_chagas_labels.csv"
def load_samitrop_data(hdf5_path, csv_path, chagas_labels_path):
    """
    Load ECG data from SaMi-Trop dataset using HDF5 and CSV files.
    
    Parameters:
    hdf5_path (str): Path to the HDF5 file containing ECG signals
    csv_path (str): Path to the CSV file with metadata
    chagas_labels_path (str): Path to the CSV file with Chagas labels
    
    Returns:
    X (np.ndarray): ECG signal data, shape (n_samples, sequence_length, 12)
    y (np.ndarray): Chagas labels, shape (n_samples,)
    """
    # Load positive Chagas labels
    labels_df = pd.read_csv(chagas_labels_path)
    positive_ids = set(labels_df['exam_id'].values)
    print(f"Loaded {len(positive_ids)} positive Chagas cases from labels file")
    
    # Load metadata
    metadata_df = pd.read_csv(csv_path)
    print(f"Loaded metadata for {len(metadata_df)} records")
    
    # Load ECG signals from HDF5
    print(f"Loading ECG signals from {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as hdf:
        # Get list of exam IDs from HDF5 file
        exam_ids = list(hdf.keys())
        print(f"Found {len(exam_ids)} records in HDF5 file")
        
        # Initialize arrays for storing data
        ecg_data = []
        labels = []
        loaded_ids = []
        
        # Loop through exams and load data
        for i, exam_id in enumerate(exam_ids):
            # Display progress
            if (i+1) % 100 == 0:
                print(f"Loaded {i+1}/{len(exam_ids)} records")
            
            try:
                # Get ECG data
                signal = np.array(hdf[exam_id])
                
                # Check if we have 12-lead ECG
                if signal.shape[1] == 12:
                    # Standardize to a fixed length if needed
                    target_length = 4000  # 10 seconds at 400Hz
                    
                    if signal.shape[0] > target_length:
                        # Trim to target length
                        signal = signal[:target_length, :]
                    elif signal.shape[0] < target_length:
                        # Pad with zeros to target length
                        padding = np.zeros((target_length - signal.shape[0], signal.shape[1]))
                        signal = np.vstack((signal, padding))
                    
                    # Add to dataset
                    ecg_data.append(signal)
                    
                    # Convert exam_id to int if needed
                    if not exam_id.isdigit():
                        continue
                        
                    exam_id_int = int(exam_id)
                    
                    # Assign label based on presence in positive_ids
                    is_chagas = 1 if exam_id_int in positive_ids else 0
                    labels.append(is_chagas)
                    loaded_ids.append(exam_id_int)
            except Exception as e:
                print(f"Error loading ECG for exam_id {exam_id}: {e}")
                continue
    
    # Convert to numpy arrays
    X = np.array(ecg_data)
    y = np.array(labels, dtype=int)
    loaded_ids = np.array(loaded_ids)
    
    print(f"Successfully loaded {len(ecg_data)} ECG records")
    print(f"ECG data shape: {X.shape}")
    print(f"Chagas positive cases: {np.sum(y)} / {len(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    
    return X, y, loaded_ids

# Preprocess ECG data
def preprocess_ecg_data(X, y):
    """
    Preprocess ECG data for model training.
    """
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Get dimensions
    n_samples, n_timesteps, n_leads = X_train.shape
    
    # Reshape for standardization (combine time and leads)
    X_train_flat = X_train.reshape(n_samples, n_timesteps * n_leads)
    X_val_flat = X_val.reshape(X_val.shape[0], n_timesteps * n_leads)
    
    # Standardize
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    
    # Reshape back to original dimensions
    X_train = X_train_flat.reshape(n_samples, n_timesteps, n_leads)
    X_val = X_val_flat.reshape(X_val.shape[0], n_timesteps, n_leads)
    
    return X_train, X_val, y_train, y_val, scaler

# Build a basic CNN model for ECG classification
def build_cnn_model(input_shape):
    """
    Build a basic CNN model for Chagas disease detection.
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', 
                    input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Second convolutional block
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Third convolutional block
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Fourth convolutional block
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), 
                tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Train CNN model for Chagas detection
def train_chagas_model(X, y, model_save_path):
    """
    Train a CNN model for Chagas disease detection.
    """
    # Preprocess data
    X_train, X_val, y_train, y_val, scaler = preprocess_ecg_data(X, y)
    
    # Save the scaler for inference
    os.makedirs(model_save_path, exist_ok=True)
    np.save(os.path.join(model_save_path, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_save_path, 'scaler_scale.npy'), scaler.scale_)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Calculate class weights if imbalanced
    class_weight = None
    if np.sum(y_train == 0) > 0 and np.sum(y_train == 1) > 0:
        weight_for_0 = 1.0
        weight_for_1 = np.sum(y_train == 0) / np.sum(y_train == 1)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"Using class weights: {class_weight}")
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_save_path, 'best_model.h5'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight
    )
    
    # Save final model
    model.save(os.path.join(model_save_path, 'final_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_history.png'))
    
    # Evaluate on validation set
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Print classification report
    print("\nValidation Set Performance:")
    print(classification_report(y_val, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'confusion_matrix.png'))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Chagas Disease Detection')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(model_save_path, 'roc_curve.png'))
    
    return model, history

# Optional: visualization of sample ECGs
def visualize_sample_ecgs(X, y, sample_indices, save_path=None):
    """
    Visualize sample ECGs from the dataset.
    """
    n_samples = len(sample_indices)
    plt.figure(figsize=(15, 3 * n_samples))
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(n_samples, 1, i+1)
        
        # Plot the 12 leads
        for lead in range(12):
            plt.plot(X[idx, :, lead] + lead * 0.5, linewidth=0.8)
        
        plt.title(f"ECG #{idx} - Chagas: {'Positive' if y[idx] == 1 else 'Negative'}")
        plt.yticks([0, 2, 4, 6, 8, 10], ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'])
        plt.xlabel('Sample')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Main function for running the entire pipeline
def main(hdf5_path, csv_path, chagas_labels_path, model_save_path):
    """
    Main function to run the Chagas detection pipeline.
    """
    # Load data
    X, y, ids = load_samitrop_data(hdf5_path, csv_path, chagas_labels_path)
    
    # Visualize some sample ECGs (optional)
    # Find some positive and negative examples
    pos_indices = np.where(y == 1)[0][:3]  # First 3 positive examples
    neg_indices = np.where(y == 0)[0][:3]  # First 3 negative examples
    sample_indices = np.concatenate([pos_indices, neg_indices])
    
    os.makedirs(model_save_path, exist_ok=True)
    # In main.py, find the line:
    visualize_sample_ecgs(X, y, sample_indices, save_path=os.path.join(model_save_path, 'sample_ecgs.png'))

    # And comment it out:
    # visualize_sample_ecgs(X, y, sample_indices, save_path=os.path.join(model_save_path, 'sample_ecgs.png'))
    
    # Train model
    model, history = train_chagas_model(X, y, model_save_path)
    
    return model, history

# If run as a script
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN model for Chagas disease detection')
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 file with ECG signals')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with metadata')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to CSV with positive Chagas labels')
    parser.add_argument('--model_path', type=str, default='./chagas_model', help='Path to save trained model')
    
    args = parser.parse_args()
    
    main(args.hdf5_path, args.csv_path, args.labels_path, args.model_path)