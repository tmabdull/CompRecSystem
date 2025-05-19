# XGBoost Model
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_processing import EXCLUDED_FEATURES_PREPROCESS

def normalize_numerical_features(df):
    """
    Normalize numerical features to 0-1 range
    Returns the scaler and the normalized dataframe
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        print("Warning: No numeric columns found for normalization")
        return None, df
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Create a copy of the dataframe to avoid modifying the original
    df_normalized = df.copy()
    
    # Apply scaling to numeric columns
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].median()))
    
    return scaler, df_normalized

def create_feature_interactions(df):
    """
    Create simple interactions between important numerical features
    """
    # List of important features that might have meaningful interactions
    important_features = ['gla', 'room_count', 'bedrooms', 'full_baths', 'half_baths']
    
    # Only use features that exist in the dataframe
    existing_features = [f for f in important_features if f in df.columns]
    
    # Create interactions between pairs of features
    for i, feat1 in enumerate(existing_features):
        for feat2 in existing_features[i+1:]:
            # Skip if either column has all NaN values
            if df[feat1].isna().all() or df[feat2].isna().all():
                continue
                
            # Create ratio features (with handling for division by zero)
            df[f'{feat1}_per_{feat2}'] = df[feat1] / df[feat2].replace(0, np.nan)
            df[f'{feat2}_per_{feat1}'] = df[feat2] / df[feat1].replace(0, np.nan)
            
            # Create product feature
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
    
    return df

def prepare_training_data(subjects_df, comps_df, properties_df):
    """
    Prepare training data for XGBoost model
    """
    print("Preparing training data...")
    
    # Get unique subject addresses (each represents one appraisal)
    all_subject_addresses = subjects_df['std_full_address'].unique()
    print(f"Total number of unique appraisals: {len(all_subject_addresses)}")
    
    # Split appraisals into train and validation sets
    train_addresses, val_addresses = train_test_split(
        all_subject_addresses, test_size=0.2, random_state=42
    )
    print(f"Training appraisals: {len(train_addresses)}, Validation appraisals: {len(val_addresses)}")
    
    # Filter dataframes based on the split
    train_subjects = subjects_df[subjects_df['std_full_address'].isin(train_addresses)]
    train_comps = comps_df[comps_df['subject_address'].isin(train_addresses)]
    train_properties = properties_df[properties_df['subject_address'].isin(train_addresses)]
    
    val_subjects = subjects_df[subjects_df['std_full_address'].isin(val_addresses)]
    val_comps = comps_df[comps_df['subject_address'].isin(val_addresses)]
    val_properties = properties_df[properties_df['subject_address'].isin(val_addresses)]
    
    print(f"Training subjects: {len(train_subjects)}, comps: {len(train_comps)}, properties: {len(train_properties)}")
    print(f"Validation subjects: {len(val_subjects)}, comps: {len(val_comps)}, properties: {len(val_properties)}")
    
    # Create positive examples (selected comps) and negative examples (non-selected properties)
    # For each subject in the training set
    train_X = []
    train_y = []
    
    for subject_address in train_addresses:
        # Get the subject property
        subject = train_subjects[train_subjects['std_full_address'] == subject_address].iloc[0]
        
        # Get comps for this subject (positive examples)
        subject_comps = train_comps[train_comps['subject_address'] == subject_address]
        
        # Get other properties for this subject (potential negative examples)
        subject_properties = train_properties[train_properties['subject_address'] == subject_address]
        
        # Process positive examples (selected comps)
        for _, comp in subject_comps.iterrows():
            # Create feature vector by comparing subject and comp
            features = create_comparison_features(subject, comp)
            train_X.append(features)
            train_y.append(1)  # 1 for selected comp
        
        # Process negative examples (non-selected properties)
        # Use properties that aren't in the comps list
        for _, prop in subject_properties.iterrows():
            # Skip if this property is already a comp (based on std_full_address)
            if any(comp['std_full_address'] == prop['std_full_address'] for _, comp in subject_comps.iterrows()):
                continue
                
            # Create feature vector by comparing subject and property
            features = create_comparison_features(subject, prop)
            train_X.append(features)
            train_y.append(0)  # 0 for non-selected property
    
    # Convert to dataframes
    train_X_df = pd.DataFrame(train_X)
    
    # Create validation data in the same way
    val_X = []
    val_y = []
    
    for subject_address in val_addresses:
        # Get the subject property
        subject = val_subjects[val_subjects['std_full_address'] == subject_address].iloc[0]
        
        # Get comps for this subject (positive examples)
        subject_comps = val_comps[val_comps['subject_address'] == subject_address]
        
        # Get other properties for this subject (potential negative examples)
        subject_properties = val_properties[val_properties['subject_address'] == subject_address]
        
        # Process positive examples (selected comps)
        for _, comp in subject_comps.iterrows():
            # Create feature vector by comparing subject and comp
            features = create_comparison_features(subject, comp)
            val_X.append(features)
            val_y.append(1)  # 1 for selected comp
        
        # Process negative examples (non-selected properties)
        for _, prop in subject_properties.iterrows():
            # Skip if this property is already a comp (based on std_full_address)
            if any(comp['std_full_address'] == prop['std_full_address'] for _, comp in subject_comps.iterrows()):
                continue
                
            # Create feature vector by comparing subject and property
            features = create_comparison_features(subject, prop)
            val_X.append(features)
            val_y.append(0)  # 0 for non-selected property
    
    # Convert to dataframes
    val_X_df = pd.DataFrame(val_X)
    
    # Create feature interactions
    train_X_df = create_feature_interactions(train_X_df)
    val_X_df = create_feature_interactions(val_X_df)
    
    # Normalize features
    scaler, train_X_normalized = normalize_numerical_features(train_X_df)
    _, val_X_normalized = normalize_numerical_features(val_X_df)
    
    return train_X_normalized, np.array(train_y), val_X_normalized, np.array(val_y), scaler

def create_comparison_features(subject, comp_or_property):
    """
    Create features that compare a subject property to a comp or potential comp
    """
    features = {}
    
    # Get all numerical features from both objects
    numerical_features = []
    for key in set(list(subject.keys()) + list(comp_or_property.keys())):
        # Skip excluded features
        address_components = set([
            'std_unit_number', 'std_street_number', 'std_street_name', 
            'std_city', 'std_province', 'std_postal_code'
        ])
        MODEL_EXCLUSION_LIST = EXCLUDED_FEATURES_PREPROCESS.union(address_components) # Adding standardized address components to exclusion list
        if key in MODEL_EXCLUSION_LIST:
            continue
            
        # Check if the feature exists in both and is numeric
        if key in subject and key in comp_or_property:
            # Try to convert to numeric to check if it's a numerical feature
            try:
                float(subject[key]) if not pd.isna(subject[key]) else 0
                float(comp_or_property[key]) if not pd.isna(comp_or_property[key]) else 0
                numerical_features.append(key)
            except (ValueError, TypeError):
                pass
    
    # Add absolute differences for numerical features
    for feature in numerical_features:
        if feature in subject and feature in comp_or_property:
            # Handle NaN values
            subj_value = float(subject[feature]) if not pd.isna(subject[feature]) else 0
            comp_value = float(comp_or_property[feature]) if not pd.isna(comp_or_property[feature]) else 0
            
            # Calculate absolute difference
            features[f'{feature}_diff'] = abs(subj_value - comp_value)
            
            # Calculate percentage difference
            if subj_value != 0:
                features[f'{feature}_pct_diff'] = abs(subj_value - comp_value) / subj_value
            else:
                features[f'{feature}_pct_diff'] = np.nan
    
    # Add raw values from comp/property
    for feature in numerical_features:
        if feature in comp_or_property:
            features[feature] = float(comp_or_property[feature]) if not pd.isna(comp_or_property[feature]) else 0
    
    return features

def train_xgboost_model(train_X, train_y, val_X, val_y):
    """
    Train XGBoost model and evaluate on validation data
    """
    print("Training XGBoost model...")
    
    # Define XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 30,  # Adjust based on your actual class ratio
        'random_state': 42
    }
    
    # Store feature names
    feature_names = list(train_X.columns)

    # Create DMatrix for XGBoost with explicit feature names
    dtrain = xgb.DMatrix(train_X, label=train_y, feature_names=feature_names)
    dval = xgb.DMatrix(val_X, label=val_y, feature_names=feature_names)
    
    # Train model with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # Store feature names in the model for later use
    model.feature_names = feature_names
    
    # Make predictions on validation set
    val_preds_prob = model.predict(dval)
    val_preds = (val_preds_prob > 0.01).astype(int)
    
    # Evaluate model
    accuracy = accuracy_score(val_y, val_preds)
    precision = precision_score(val_y, val_preds, zero_division=0)
    recall = recall_score(val_y, val_preds)
    f1 = f1_score(val_y, val_preds)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    
    return model, val_preds_prob, accuracy, precision, recall, f1
