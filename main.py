import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Step 1: Load CSV data
def load_csv_data():
    print("Step 1: Loading CSV data")
    exams_df = pd.read_csv('exams.csv')
    labels_df = pd.read_csv('samitrop_chagas_labels.csv')
    
    print(f"Exams data: {len(exams_df)} rows")
    print(f"Labels data: {len(labels_df)} rows")
    
    # Display sample data
    print("\nExams data sample:")
    print(exams_df.head(3))
    print("\nLabels data sample:")
    print(labels_df.head(3))
    
    # Merge dataframes on exam_id
    merged_df = pd.merge(exams_df, labels_df, on='exam_id', how='inner')
    print(f"\nTotal matched records: {len(merged_df)}")
    
    # Check class balance
    chagas_count = merged_df['chagas'].sum()
    print(f"Chagas positive: {chagas_count} ({chagas_count/len(merged_df)*100:.2f}%)")
    print(f"Chagas negative: {len(merged_df) - chagas_count} ({(1-chagas_count/len(merged_df))*100:.2f}%)")
    
    return merged_df

# Step 2: Load ECG data from HDF5 file
def load_ecg_data(merged_df):
    print("\nStep 2: Loading ECG data from HDF5 file")
    
    if not os.path.exists('exams.hdf5'):
        raise FileNotFoundError("The file 'exams.hdf5' was not found in the current directory.")
    
    # Open HDF5 file
    with h5py.File('exams.hdf5', 'r') as hdf:
        # Print HDF5 structure information
        print("HDF5 file structure:")
        print_hdf5_structure(hdf)
        
        # Get list of available exam_ids in the HDF5 file
        # We need to handle different possible HDF5 structures
        if isinstance(hdf, h5py.Group):
            # Case 1: HDF5 has direct keys as exam_ids
            available_ids = list(hdf.keys())
            get_ecg = lambda id: np.array(hdf[id])
        elif 'data' in hdf:
            # Case 2: HDF5 has structured datasets
            # This is a placeholder - adjust based on actual structure
            print("HDF5 has a 'data' dataset. Need to determine how exam_ids map to this dataset.")
            # This is just an example, adapt to your file structure
            available_ids = [str(i) for i in range(len(hdf['data']))]
            get_ecg = lambda id: np.array(hdf['data'][int(id)])
        else:
            # Inspect and adapt to the actual structure
            print("WARNING: Unknown HDF5 structure. Trying to infer structure...")
            if len(list(hdf.keys())) > 0:
                first_key = list(hdf.keys())[0]
                print(f"First key in HDF5: {first_key}")
                if isinstance(hdf[first_key], h5py.Group):
                    # Nested structure
                    available_ids = []
                    for group_key in hdf.keys():
                        available_ids.extend([f"{group_key}/{item}" for item in hdf[group_key].keys()])
                    get_ecg = lambda id: np.array(hdf[id.split('/')[0]][id.split('/')[1]])
                else:
                    # Flat structure but with different naming
                    available_ids = list(hdf.keys())
                    get_ecg = lambda id: np.array(hdf[id])
            else:
                raise ValueError("Unable to determine HDF5 structure. Empty file or incompatible format.")
        
        print(f"Total ECGs in HDF5: {len(available_ids)}")
        print(f"First few available IDs: {available_ids[:5] if len(available_ids) >= 5 else available_ids}")
        
        # Filter merged_df to only include exam_ids available in the HDF5
        # Try both string and integer formats
        exam_ids_int = merged_df['exam_id'].tolist()
        exam_ids_str = [str(id) for id in exam_ids_int]
        
        # Check if any string IDs match
        valid_ids_str = [id for id in exam_ids_str if id in available_ids]
        if valid_ids_str:
            print(f"Found {len(valid_ids_str)} matching string IDs")
            valid_ids = valid_ids_str
            id_format = "string"
        else:
            # If no string matches, try different formats (e.g., zero-padded)
            print("No direct string matches. Trying alternative formats...")
            # Try zero-padded IDs (e.g., "001" instead of "1")
            padded_ids = [str(id).zfill(6) for id in exam_ids_int]  # Adjust padding as needed
            valid_ids_padded = [id for id in padded_ids if id in available_ids]
            if valid_ids_padded:
                print(f"Found {len(valid_ids_padded)} matching padded IDs")
                valid_ids = valid_ids_padded
                id_format = "padded"
                # Map back to original exam_ids
                id_mapping = {padded: orig for padded, orig in zip(padded_ids, exam_ids_str)}
            else:
                # If still no matches, look for IDs within available_ids
                print("No padded matches. Trying substring matching...")
                valid_ids = []
                id_format = "substring"
                for avail_id in available_ids:
                    # Check if any of our exam_ids are contained in avail_id
                    for exam_id in exam_ids_str:
                        if exam_id in avail_id:
                            valid_ids.append(avail_id)
                            break
                print(f"Found {len(valid_ids)} substring matches")
        
        if not valid_ids:
            # Last resort: Check if numerical indices are used instead of IDs
            print("No ID matches found. Trying numerical indices...")
            # This assumes HDF5 uses numerical indices and our exam_ids map to these indices
            # This is highly file-specific and may need adjustment
            all_indices = list(range(len(available_ids)))
            # Try to use exam_ids as indices if they're in a valid range
            valid_indices = [i for i in exam_ids_int if i < len(available_ids)]
            if valid_indices:
                print(f"Using {len(valid_indices)} exam_ids as numerical indices")
                valid_ids = [available_ids[i] for i in valid_indices]
                id_format = "index"
                # Map HDF5 keys to original exam_ids
                id_mapping = {available_ids[i]: str(exam_id) for i, exam_id in enumerate(valid_indices)}
            else:
                raise ValueError("No matching exam_ids found in HDF5 file. Cannot proceed.")
        
        print(f"Valid ECGs with labels: {len(valid_ids)}")
        if len(valid_ids) == 0:
            raise ValueError("No matching exam_ids found between CSV and HDF5. Cannot proceed.")
        
        # Print sample of valid IDs
        print(f"First few valid IDs: {valid_ids[:5] if len(valid_ids) >= 5 else valid_ids}")
        
        # Initialize arrays to store ECG data and labels
        ecg_data = []
        labels = []
        original_ids = []
        
        # Load ECG data for each valid exam_id
        print("Loading ECG data...")
        for i, id in enumerate(valid_ids):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(valid_ids)} ECGs loaded")
            
            # Get the original exam_id to find the label
            if id_format == "string":
                original_id = int(id)
            elif id_format == "padded":
                original_id = int(id_mapping[id])
            elif id_format == "substring":
                # Find matching exam_id
                for exam_id in exam_ids_str:
                    if exam_id in id:
                        original_id = int(exam_id)
                        break
            elif id_format == "index":
                original_id = int(id_mapping[id])
            else:
                original_id = int(id)
            
            # Get the binary label (1 for Chagas, 0 for non-Chagas)
            label_row = merged_df.loc[merged_df['exam_id'] == original_id]
            if len(label_row) == 0:
                print(f"  Warning: No label found for ID {original_id}. Skipping.")
                continue
                
            label = label_row['chagas'].values[0]
            
            try:
                # Get ECG data from HDF5
                ecg = get_ecg(id)
                
                # Check if we need to reshape the data
                if len(ecg.shape) == 1:
                    # Single-lead ECG needs reshaping
                    print(f"  Single-dimension ECG found with shape {ecg.shape}. Reshaping...")
                    # Assume it's 12 leads concatenated, reshape to (n_samples, 12)
                    n_samples = ecg.shape[0] // 12
                    ecg = ecg.reshape(n_samples, 12)
                elif len(ecg.shape) > 2:
                    # Too many dimensions, try to flatten to 2D
                    print(f"  Multi-dimension ECG found with shape {ecg.shape}. Reshaping...")
                    # This is a placeholder - adjust based on actual data
                    ecg = ecg.reshape(ecg.shape[0], -1)
                
                # Print the shape of the first ECG to help debug
                if i == 0:
                    print(f"  First ECG shape: {ecg.shape}")
                    # Based on shape, determine if we need to transpose
                    # Typical shape is (time_points, leads) or (leads, time_points)
                    if ecg.shape[1] != 12 and ecg.shape[0] == 12:
                        print("  Transposing ECG data to get (time_points, leads) format")
                        ecg = ecg.T
                        print(f"  Transposed shape: {ecg.shape}")
                
                # Verify shape is reasonable (time_points, 12_leads)
                if ecg.shape[1] != 12:
                    print(f"  Warning: ECG for ID {id} has shape {ecg.shape}, expected 12 leads in dimension 1")
                    if ecg.shape[0] == 12:
                        # Transpose if needed
                        ecg = ecg.T
                        print(f"  Transposed shape: {ecg.shape}")
                
                # Add to lists
                ecg_data.append(ecg)
                labels.append(label)
                original_ids.append(original_id)
            except Exception as e:
                print(f"  Error loading ECG for ID {id}: {e}")
                continue
        
        if len(ecg_data) == 0:
            raise ValueError("No ECG data was successfully loaded. Cannot proceed.")
        
        print(f"Successfully loaded {len(ecg_data)} ECGs")
        
        # Check if ECGs have consistent shapes
        shapes = [ecg.shape for ecg in ecg_data]
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            print(f"Warning: Inconsistent ECG shapes detected. Found {len(unique_shapes)} different shapes.")
            print(f"Shape counts: {[(shape, shapes.count(shape)) for shape in unique_shapes]}")
            
            # Find the most common shape
            most_common_shape = max(unique_shapes, key=shapes.count)
            print(f"Using most common shape: {most_common_shape}")
            
            # Filter to keep only ECGs with the most common shape
            filtered_data = []
            filtered_labels = []
            filtered_ids = []
            for ecg, label, eid in zip(ecg_data, labels, original_ids):
                if ecg.shape == most_common_shape:
                    filtered_data.append(ecg)
                    filtered_labels.append(label)
                    filtered_ids.append(eid)
            
            ecg_data = filtered_data
            labels = filtered_labels
            original_ids = filtered_ids
            print(f"After filtering: {len(ecg_data)} ECGs with consistent shape")
        
        # Convert lists to numpy arrays, ensuring consistent shapes
        if len(ecg_data) > 0:
            # Ensure all ECGs have the same shape by padding/truncating if needed
            time_points = max(ecg.shape[0] for ecg in ecg_data)
            n_leads = ecg_data[0].shape[1]  # Assuming all have same number of leads after filtering
            
            X = np.zeros((len(ecg_data), time_points, n_leads))
            for i, ecg in enumerate(ecg_data):
                # Pad or truncate to the same length
                if ecg.shape[0] < time_points:
                    # Pad with zeros
                    X[i, :ecg.shape[0], :] = ecg
                else:
                    # Truncate
                    X[i, :, :] = ecg[:time_points, :]
                    
            y = np.array(labels)
            
            print(f"Final ECG data shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            
            return X, y, original_ids
        else:
            raise ValueError("No valid ECG data after filtering. Cannot proceed.")

# Helper function to print HDF5 structure
def print_hdf5_structure(hdf_file, indent=0):
    """Recursively print the HDF5 file structure."""
    try:
        for key in hdf_file.keys():
            if isinstance(hdf_file[key], h5py.Group):
                print("  " * indent + f"Group: {key}")
                print_hdf5_structure(hdf_file[key], indent + 1)
            else:
                print("  " * indent + f"Dataset: {key}, Shape: {hdf_file[key].shape}")
    except:
        print("  " * indent + "Error exploring HDF5 structure")

# Step 3: Preprocess ECG data
def preprocess_data(X, y):
    print("\nStep 3: Preprocessing ECG data")
    
    # Check if we have enough data to split
    if len(X) < 5:
        raise ValueError(f"Not enough samples to split: {len(X)} samples. Need at least 5.")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Normalize each ECG to have zero mean and unit variance
    # We normalize each lead separately
    for i in range(X_train.shape[0]):
        for lead in range(X_train.shape[2]):
            # Avoid division by zero
            std = np.std(X_train[i, :, lead])
            if std == 0:
                X_train[i, :, lead] = 0  # Just set to zeros if constant
            else:
                X_train[i, :, lead] = (X_train[i, :, lead] - np.mean(X_train[i, :, lead])) / std
    
    for i in range(X_test.shape[0]):
        for lead in range(X_test.shape[2]):
            # Avoid division by zero
            std = np.std(X_test[i, :, lead])
            if std == 0:
                X_test[i, :, lead] = 0  # Just set to zeros if constant
            else:
                X_test[i, :, lead] = (X_test[i, :, lead] - np.mean(X_test[i, :, lead])) / std
    
    return X_train, X_test, y_train, y_test

# Step 4: Visualize sample ECGs
def visualize_samples(X, y, original_ids, n_samples=3):
    print("\nStep 4: Visualizing sample ECGs")
    
    # Get indices of Chagas positive and negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # Check if we have samples of both classes
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        print("Cannot visualize: missing examples from one or both classes")
        return
    
    # Plot Chagas positive samples
    plt.figure(figsize=(15, 10))
    for i in range(min(n_samples, len(pos_indices))):
        idx = pos_indices[i]
        plt.subplot(2, n_samples, i+1)
        
        # Plot all leads
        for lead in range(X.shape[2]):
            plt.plot(X[idx, :, lead], alpha=0.7, linewidth=0.5)
        
        plt.title(f"Chagas Positive (ID: {original_ids[idx]})")
        plt.ylim(-5, 5)  # Standardized scale
    
    # Plot Chagas negative samples
    for i in range(min(n_samples, len(neg_indices))):
        idx = neg_indices[i]
        plt.subplot(2, n_samples, i+n_samples+1)
        
        # Plot all leads
        for lead in range(X.shape[2]):
            plt.plot(X[idx, :, lead], alpha=0.7, linewidth=0.5)
        
        plt.title(f"Chagas Negative (ID: {original_ids[idx]})")
        plt.ylim(-5, 5)  # Standardized scale
    
    plt.tight_layout()
    plt.savefig('sample_ecgs.png')
    plt.close()
    print("Sample ECGs saved to 'sample_ecgs.png'")

# Step 5: Build 1D CNN model
def build_model(input_shape):
    print("\nStep 5: Building 1D CNN model")
    
    model = Sequential([
        # First Conv layer
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # Second Conv layer
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Third Conv layer
        Conv1D(filters=256, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Print model summary
    model.summary()
    return model

# Step 6: Train model
def train_model(model, X_train, y_train, X_test, y_test):
    print("\nStep 6: Training model")
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history saved to 'training_history.png'")
    
    return model, history

# Step 7: Evaluate model
def evaluate_model(model, X_test, y_test):
    print("\nStep 7: Evaluating model")
    
    # Predict on test set
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    print("ROC curve saved to 'roc_curve.png'")
    
    return y_pred, y_pred_prob, roc_auc

# Step 8: Analyze model predictions
def analyze_predictions(merged_df, original_ids, y_test, y_pred, y_pred_prob):
    print("\nStep 8: Analyzing model predictions")
    
    # Convert original_ids to appropriate format for test set
    test_ids = original_ids[-len(y_test):]  # Assuming test set is last portion
    
    # Create DataFrame with test set results
    test_results = pd.DataFrame({
        'exam_id': test_ids,
        'true_label': y_test,
        'predicted_prob': y_pred_prob.flatten(),
        'predicted_label': y_pred
    })
    
    # Add demographics from merged_df
    test_results['exam_id'] = test_results['exam_id'].astype(int)
    test_results = pd.merge(test_results, 
                           merged_df[['exam_id', 'age', 'is_male', 'normal_ecg', 'death']], 
                           on='exam_id', how='left')
    
    # False positives (predicted Chagas but actually negative)
    false_pos = test_results[(test_results['predicted_label'] == 1) & (test_results['true_label'] == 0)]
    print(f"False positives: {len(false_pos)}")
    if len(false_pos) > 0:
        print("Demographics of false positives:")
        print(f"  Average age: {false_pos['age'].mean():.1f}")
        print(f"  Percentage male: {false_pos['is_male'].mean()*100:.1f}%")
        print(f"  Percentage with normal ECG: {false_pos['normal_ecg'].mean()*100:.1f}%")
    
    # False negatives (predicted negative but actually Chagas)
    false_neg = test_results[(test_results['predicted_label'] == 0) & (test_results['true_label'] == 1)]
    print(f"False negatives: {len(false_neg)}")
    if len(false_neg) > 0:
        print("Demographics of false negatives:")
        print(f"  Average age: {false_neg['age'].mean():.1f}")
        print(f"  Percentage male: {false_neg['is_male'].mean()*100:.1f}%")
        print(f"  Percentage with normal ECG: {false_neg['normal_ecg'].mean()*100:.1f}%")
    
    # Save predictions to CSV
    test_results.to_csv('chagas_predictions.csv', index=False)
    print("Predictions saved to 'chagas_predictions.csv'")

# Main function
def main():
    print("ECG-Based Chagas Disease Detection")
    print("==================================")
    
    try:
        # Step 1: Load CSV data
        merged_df = load_csv_data()
        
        # Step 2: Load ECG data
        X, y, original_ids = load_ecg_data(merged_df)
        
        # Step 3: Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        
        # Step 4: Visualize sample ECGs
        visualize_samples(X, y, original_ids)
        
        # Step 5 & 6: Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])  # (time steps, leads)
        model = build_model(input_shape)
        model, history = train_model(model, X_train, y_train, X_test, y_test)
        
        # Step 7: Evaluate model
        y_pred, y_pred_prob, roc_auc = evaluate_model(model, X_test, y_test)
        
        # Step 8: Analyze predictions
        analyze_predictions(merged_df, original_ids, y_test, y_pred, y_pred_prob)
        
        # Save model
        model.save('chagas_detection_model.h5')
        print("\nModel saved to 'chagas_detection_model.h5'")
        
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that all files (exams.csv, samitrop_chagas_labels.csv, exams.hdf5) are in the current directory")
        print("2. Verify that the HDF5 file structure matches what the script expects")
        print("3. Run with smaller batch size if memory errors occur")
        print("4. Ensure TensorFlow and other dependencies are properly installed")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()