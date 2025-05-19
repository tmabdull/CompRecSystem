# Recommendation System
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import datetime
import pickle
import os
from sklearn.model_selection import train_test_split

from xgboost_model import create_comparison_features, create_feature_interactions, prepare_training_data, train_xgboost_model
from data_processing import load_and_process_data

def explain_recommendation(model, subject, candidate_property, feature_names):
    """
    Generate SHAP explanation for a single recommendation
    """
    # Create comparison features
    features = create_comparison_features(subject, candidate_property)
    
    # Convert to dataframe
    features_df = pd.DataFrame([features])
    
    # Create feature interactions
    features_df = create_feature_interactions(features_df)
    
    # Ensure feature order matches training
    features_df = features_df[feature_names]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(features_df)
    
    # Generate explanation text
    explanation = []
    for i in np.argsort(-np.abs(shap_values[0])):
        feature = feature_names[i]
        value = features_df.iloc[0][feature]
        shap_value = shap_values[0][i]
        
        if "diff" in feature or "pct_diff" in feature:
            base_feature = feature.replace("_diff", "").replace("_pct_diff", "")
            explanation.append(
                f"{base_feature} similarity: {value:.2f} (Impact: {shap_value:.2f})"
            )
    
    return explanation[:3]  # Return top 3 factors

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def recommend_comps(model, scaler, subject, candidate_properties, top_n=3, max_distance_km=5.0):
    """
    Recommend top N comparable properties with SHAP explanations for a given subject property
    
    Args:
        model: Trained XGBoost model
        scaler: Feature scaler used during training
        subject: Subject property (Series or dict)
        candidate_properties: DataFrame of candidate properties
        top_n: Number of recommendations to return (default: 3)
        max_distance_km: Maximum distance in kilometers to consider (default: 5.0)
    
    Returns:
        DataFrame with top N recommended properties and explanations
    """
    # Filter by distance if distance information is available
    filtered_candidates = candidate_properties.copy()
    
    # If we have latitude/longitude coordinates
    if all(col in filtered_candidates.columns for col in ['latitude', 'longitude']) and \
       all(col in subject for col in ['latitude', 'longitude']):
        # Calculate distance using Haversine formula
        filtered_candidates['distance_km'] = filtered_candidates.apply(
            lambda row: calculate_distance(
                subject['latitude'], subject['longitude'],
                row['latitude'], row['longitude']
            ),
            axis=1
        )
        # Filter by calculated distance
        filtered_candidates = filtered_candidates[filtered_candidates['distance_km'] <= max_distance_km]
    
    # If we have distance_to_subject field in the candidates
    elif 'distance_to_subject' in filtered_candidates.columns:
        # Ensure distance is in km and is numeric
        filtered_candidates = filtered_candidates[
            pd.to_numeric(filtered_candidates['distance_to_subject'], errors='coerce') <= max_distance_km
        ]
    
    # If no candidates remain after distance filtering, return empty DataFrame
    if len(filtered_candidates) == 0:
        print(f"Warning: No candidates within {max_distance_km} km of subject property")
        return pd.DataFrame()
    
    # Create comparison features for each candidate property
    candidate_features = []
    for _, prop in filtered_candidates.iterrows():
        features = create_comparison_features(subject, prop)
        candidate_features.append(features)
    
    # Convert to dataframe
    candidate_features_df = pd.DataFrame(candidate_features)
    
    # Create feature interactions
    candidate_features_df = create_feature_interactions(candidate_features_df)

    # Get the feature names used during training, in the same order
    train_features = model.feature_names

    # Ensure all training features exist in candidate features
    for feature in train_features:
        if feature not in candidate_features_df.columns:
            candidate_features_df[feature] = 0  # Add missing features with default values
    
    # Select only the features used in training and in the same order
    candidate_features_df = candidate_features_df[train_features]
    
    # Normalize features using the same scaler used during training
    candidate_features_normalized = candidate_features_df.copy()
    
    # Apply scaling if there are numeric columns
    if hasattr(scaler, 'feature_names_in_'):
        numeric_cols = candidate_features_df.columns
        candidate_features_normalized = pd.DataFrame(
            scaler.transform(candidate_features_df.fillna(candidate_features_df.median())),
            columns=numeric_cols
        )
    
    # Create DMatrix
    dcandidate = xgb.DMatrix(candidate_features_normalized)
    
    # Get predictions
    candidate_scores = model.predict(dcandidate)
    
    # Add scores to candidate properties
    filtered_candidates['comp_score'] = candidate_scores
    
    # Sort by score and get top N
    top_comps = filtered_candidates.sort_values('comp_score', ascending=False).head(top_n)
    
    # Add explanations
    top_comps['explanation'] = top_comps.apply(
        lambda row: explain_recommendation(model, subject, row, feature_names=model.feature_names),
        axis=1
    )

    return top_comps

def evaluate_recommendations(model, scaler, val_subjects, val_comps, val_properties):
    """
    Evaluate the recommendation system on validation data
    
    Args:
        model: Trained XGBoost model
        scaler: Feature scaler used during training
        val_subjects: DataFrame of validation subject properties
        val_comps: DataFrame of validation comp properties
        val_properties: DataFrame of validation candidate properties
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating recommendation system...")
    
    # Initialize metrics
    total_subjects = 0
    total_recommendations = 0
    total_hits = 0
    precision_at_3_sum = 0
    
    # Process each subject property
    for _, subject in val_subjects.iterrows():
        subject_address = subject['std_full_address']
        
        # Get actual comps for this subject
        actual_comps = val_comps[val_comps['subject_address'] == subject_address]
        
        # Skip if no actual comps
        if len(actual_comps) == 0:
            continue
            
        # Get candidate properties for this subject
        candidates = val_properties[val_properties['subject_address'] == subject_address]
        
        # Skip if no candidates
        if len(candidates) == 0:
            continue
            
        # Get recommended comps
        recommended_comps = recommend_comps(model, scaler, subject, candidates, top_n=3)
        
        # Skip if no recommendations
        if len(recommended_comps) == 0:
            continue
            
        # Count hits (recommended comps that match actual comps)
        hits = 0
        for _, rec_comp in recommended_comps.iterrows():
            # Check if this recommendation matches any actual comp
            for _, act_comp in actual_comps.iterrows():
                # Match based on standardized full address
                if ('std_full_address' in rec_comp and 'std_full_address' in act_comp and 
                    rec_comp['std_full_address'] == act_comp['std_full_address']):
                    hits += 1
                    break
                # Fallback to regular address if standardized not available
                elif 'id' in rec_comp and 'id' in act_comp and rec_comp['id'] == act_comp['id']:
                    hits += 1
                    break
        
        # Update metrics
        total_subjects += 1
        total_recommendations += len(recommended_comps)
        total_hits += hits
        precision_at_3 = hits / min(3, len(recommended_comps))
        precision_at_3_sum += precision_at_3
    
    # Calculate aggregate metrics
    if total_subjects > 0:
        avg_precision_at_3 = precision_at_3_sum / total_subjects
        overall_precision = total_hits / total_recommendations if total_recommendations > 0 else 0
        recall = total_hits / (total_subjects * 3) if total_subjects > 0 else 0
        
        print(f"Evaluated on {total_subjects} subject properties")
        print(f"Average Precision@3: {avg_precision_at_3:.4f}")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        
        return {
            'num_subjects': total_subjects,
            'avg_precision_at_3': avg_precision_at_3,
            'overall_precision': overall_precision,
            'overall_recall': recall
        }
    else:
        print("No valid subjects for evaluation")
        return {
            'num_subjects': 0,
            'avg_precision_at_3': 0,
            'overall_precision': 0,
            'overall_recall': 0
        }


def load_latest_model_and_scaler():
    """
    Load the most recent model and scaler based on timestamp in filename
    """
    models_dir = "./data/models/"
    
    # Get all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.json')]
    
    if not model_files:
        raise FileNotFoundError("No model files found in ./data/models/")
    
    # Sort by timestamp (assuming filename format includes timestamp)
    # The format is comp_recommendation_model_YYYYMMDD_HHMMSS_accuracy_precision_recall_f1.json
    latest_model_file = sorted(model_files)[-1]
    
    # Extract base filename without extension
    base_filename = latest_model_file.replace('.json', '')
    
    # Construct paths
    model_path = os.path.join(models_dir, latest_model_file)
    scaler_path = os.path.join(models_dir, f"{base_filename}_scaler.pkl")
    feature_names_path = os.path.join(models_dir, f"{base_filename}_features.pkl")
    
    # Check if scaler and feature names files exist
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    
    # Load model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature names
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    # Add feature names to model
    model.feature_names = feature_names
    
    print(f"Loaded model and scaler from {model_path}")
    return model, scaler

# Main execution
def train_and_evaluate_xgboost(force_reprocessing=False):
    # Load and process data
    if (os.path.exists("./data/processed/processed_subjects.pkl") and 
        os.path.exists("./data/processed/processed_comps.pkl") and 
        os.path.exists("./data/processed/processed_properties.pkl") and 
        force_reprocessing == False):
        # Load directly from pkl
        print("Appraisal df pkl files found! Loading...")
        subjects_df = pd.read_pickle("./data/processed/processed_subjects.pkl")
        comps_df = pd.read_pickle("./data/processed/processed_comps.pkl")
        properties_df = pd.read_pickle("./data/processed/processed_properties.pkl")
    else:
        # Create dfs by processing the raw data
        if not force_reprocessing:
            print("Could not find appraisal df pkl files.")
        print("Creating dfs by processing the raw data...")
        subjects_df, comps_df, properties_df = load_and_process_data()
    print("Appraisal dfs loaded successfully")
    
    # Prepare training data
    train_X, train_y, val_X, val_y, scaler = prepare_training_data(subjects_df, comps_df, properties_df)

    # Train XGBoost model
    model, val_preds_prob, accuracy, precision, recall, f1 = train_xgboost_model(train_X, train_y, val_X, val_y)

    # Save model, scaler, and feature_names for later use
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"comp_recommendation_model_{timestamp}_{accuracy:.2f}_{precision:.2f}_{recall:.2f}_{f1:.2f}"
        
    model_path = f"./data/models/{base_filename}.json"
    model.save_model(model_path)

    scaler_path = f"./data/models/{base_filename}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save feature names separately for easy access across sessions
    feature_names_path = f"./data/models/{base_filename}_features.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(model.feature_names, f)

    print(f"Model, scaler, and feature_names saved with timestamp {timestamp}")
        
    # Get validation subjects, comps, and properties
    all_subject_addresses = subjects_df['std_full_address'].unique()
    train_addresses, val_addresses = train_test_split(
        all_subject_addresses, test_size=0.2, random_state=42
    )

    val_subjects = subjects_df[subjects_df['std_full_address'].isin(val_addresses)]
    val_comps = comps_df[comps_df['subject_address'].isin(val_addresses)]
    val_properties = properties_df[properties_df['subject_address'].isin(val_addresses)]

    # Evaluate recommendations on validation set
    evaluation_metrics = evaluate_recommendations(model, scaler, val_subjects, val_comps, val_properties)
    print(evaluation_metrics)

    # Example of recommending comps for a specific subject property
    if len(val_subjects) > 0:
        example_subject = val_subjects.iloc[0]
        example_candidates = val_properties[val_properties['subject_address'] == example_subject['std_full_address']]
        
        print("\nExample Recommendations:")
        print(f"Subject Property: {example_subject['std_full_address']}")
        
        recommended_comps = recommend_comps(model, scaler, example_subject, example_candidates, top_n=3)
        
        if len(recommended_comps) > 0:
            print("\nRecommended comps:")
            display_cols = ['std_full_address', 'comp_score']
            display_cols = [col for col in display_cols if col in recommended_comps.columns]
            print(recommended_comps[display_cols])
            
            # Show actual comps for comparison
            actual_comps = val_comps[val_comps['subject_address'] == example_subject['std_full_address']]
            if len(actual_comps) > 0:
                print("\nActual comps selected by appraiser:")
                display_cols = [col for col in display_cols if col in actual_comps.columns]
                print(actual_comps[display_cols])

# If running as a script
if __name__ == "__main__":
    train_and_evaluate_xgboost()
