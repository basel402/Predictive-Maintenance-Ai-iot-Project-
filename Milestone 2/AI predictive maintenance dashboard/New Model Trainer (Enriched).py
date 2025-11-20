import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------

# Define file paths
DATA_FILE = 'ai4i2020_enriched.csv' # <-- Use the new enriched file
MODEL_FILE = 'voting_model_ai4i_enriched.joblib' # <-- New model name
SCALER_FILE = 'scaler_ai4i_enriched.joblib' # <-- New scaler name

# Define column groups
TARGET_COL = 'Machine failure'
FAILURE_FLAGS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
NUMERICAL_COLS = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
]
CATEGORICAL_COLS = ['Type']

# New columns in the enriched file that are NOT for training
METADATA_COLS = [
    'UDI', 
    'Product ID',
    'event_timestamp',
    'downtime_hours',
    'repair_cost',
    'repair_notes'
]

# ------------------------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------------------------

def main():
    """
    Main function to load, preprocess, train, and save the model.
    """
    start_time = time.time()
    
    # --- 1. Load Data ---
    print(f"--- Loading data from '{DATA_FILE}' ---")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found.")
        print("Please make sure you have run 'create_enriched_data.py' first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data loaded successfully. Shape: {df.shape}")

    # --- 2. Preprocessing & Splitting ---
    print("--- Preprocessing and splitting data ---")
    
    # Define features (X) and target (y)
    # We drop the target, all failure flags, and all new metadata columns
    try:
        X = df.drop(columns=[TARGET_COL] + FAILURE_FLAGS + METADATA_COLS, 
                    errors='ignore')
        y = df[TARGET_COL]
    except KeyError as e:
        print(f"Error: Missing expected column. {e}")
        return

    # Check for missing columns needed for training
    required_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    missing_cols = [col for col in required_cols if col not in X.columns]
    if missing_cols:
        print(f"Error: Missing required training columns: {missing_cols}")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Define Preprocessing Pipeline ---
    print("--- Defining preprocessing pipeline ---")
    
    # Numerical transformer: scaling
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical transformer: one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a column transformer to apply different transforms to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='passthrough'
    )

    # --- 4. Define Models ---
    print("--- Defining models ---")
    
    # Define individual classifiers with optimized parameters
    # These parameters are commonly effective for this dataset
    rf_clf = RandomForestClassifier(
        random_state=42, 
        n_estimators=150, 
        max_depth=20, 
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    xgb_clf = xgb.XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss',
        n_estimators=120,
        learning_rate=0.1,
        max_depth=5
    )
    
    lgb_clf = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=120,
        learning_rate=0.1,
        num_leaves=31
    )

    # Define the Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_clf),
            ('xgb', xgb_clf),
            ('lgb', lgb_clf)
        ],
        voting='soft' # Use probabilities for better performance
    )

    # --- 5. Create Full SMOTE-enabled Pipeline ---
    print("--- Creating final pipeline with SMOTE ---")
    
    # We use ImbPipeline to ensure SMOTE is only applied to training data
    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', voting_clf)
    ])

    # --- 6. Train the Model ---
    print("--- Training the final model... (This may take a few minutes) ---")
    
    full_pipeline.fit(X_train, y_train)
    
    print("Model training complete.")
    
    # --- 7. Save the Preprocessor (Scaler) and Model ---
    
    # We save the *preprocessor* step from the pipeline, not just a scaler
    # This ensures OHE and scaling are applied correctly in the app
    preprocessor_to_save = full_pipeline.named_steps['preprocessor']
    
    print(f"--- Saving preprocessor to '{SCALER_FILE}' ---")
    joblib.dump(preprocessor_to_save, SCALER_FILE)
    
    # We save the *classifier* step (the VotingClassifier)
    model_to_save = full_pipeline.named_steps['classifier']
    
    print(f"--- Saving model to '{MODEL_FILE}' ---")
    joblib.dump(model_to_save, MODEL_FILE)
    
    # --- 8. Evaluate and Print Report ---
    print("--- Evaluating model on test set ---")
    
    # Must apply *only* preprocessing (no SMOTE) to test data
    X_test_processed = preprocessor_to_save.transform(X_test)
    y_pred = model_to_save.predict(X_test_processed)
    
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred))
    
    end_time = time.time()
    print(f"\n--- Process finished in {end_time - start_time:.2f} seconds ---")
    print(f"Successfully created '{MODEL_FILE}' and '{SCALER_FILE}'.")

if __name__ == "__main__":
    main()
