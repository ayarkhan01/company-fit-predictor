import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, is_training=True):
    """
    Preprocess the data for both training and prediction
    """
    # Select relevant columns for training
    columns = ['Employee Estimate', 'Employees on Professional Networks', 'Employee Growth (Monthly)', 
               'Employee Growth (Quarterly)', 'Employee Growth (6 months)', 'Employee Growth (Annual)', 
               'Headquarters', 'Year Founded', 'Last Funding Amount', 'Last Funding Date', 
               'Total Funding Amount', 'Total Funding Rounds', 'Business Model', 'Last Touch Date', 
               'Last Pipeline Decline Date']
    
    # Add Rank column for training data
    if is_training:
        columns.append('Rank')
    
    # Only select columns that exist in the dataframe
    available_columns = [col for col in columns if col in df.columns]
    df_processed = df[available_columns].copy()
    
    # US location check
    us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    
    def is_us_location(location):
        if pd.isna(location):
            return 0.5
        return int(any(state in str(location) for state in us_states))
    
    if 'Headquarters' in df_processed.columns:
        df_processed['is_us'] = df_processed['Headquarters'].apply(is_us_location)
    else:
        df_processed['is_us'] = 0.5  # Default for missing column
    
    # Calculate company age
    current_year = datetime.now().year
    if 'Year Founded' in df_processed.columns:
        df_processed['company_age'] = current_year - df_processed['Year Founded']
    else:
        df_processed['company_age'] = 10  # Default age
    
    # Check for tech-enabled sectors
    tech_enabled_sectors = ['Software', 'Software Enabled', 'Tech', 'Technology']
    
    def is_tech_enabled(sector):
        if pd.isna(sector):
            return 0
        return int(str(sector) in tech_enabled_sectors)
    
    if 'Business Model' in df_processed.columns:
        df_processed['is_tech_enabled'] = df_processed['Business Model'].apply(is_tech_enabled)
    else:
        df_processed['is_tech_enabled'] = 0  # Default for missing column
    
    # Adjust rank for training (convert 1-10 scale to 0-9 for better ML performance)
    if is_training and 'Rank' in df_processed.columns:
        df_processed['Rank'] -= 1
    
    # Clean data - drop original columns that were transformed
    columns_to_drop = ['Year Founded', 'Headquarters', 'Business Model']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
    # Remove rows with missing target variable for training
    if is_training and 'Rank' in df_processed.columns:
        df_processed = df_processed.dropna(subset=['Rank'])
    
    # Convert 'Employee Growth' columns to float
    growth_columns = ['Employee Growth (Monthly)', 'Employee Growth (Quarterly)', 
                     'Employee Growth (6 months)', 'Employee Growth (Annual)']
    
    for col in growth_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.replace('%', '').replace('nan', np.nan)
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        else:
            df_processed[col] = 0.0  # Default for missing growth columns
    
    # Process date columns
    today = pd.Timestamp.now().tz_localize(None)
    
    def safe_to_timestamp(x):
        if pd.isna(x):
            return np.nan
        try:
            return pd.to_datetime(x).tz_localize(None)
        except:
            return np.nan
    
    date_columns = ['Last Funding Date', 'Last Touch Date', 'Last Pipeline Decline Date']
    for col in date_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(safe_to_timestamp)
            df_processed[f'Days Since {col}'] = (today - df_processed[col]).dt.days.replace({pd.NaT: np.nan})
        else:
            df_processed[f'Days Since {col}'] = np.nan  # Default for missing date columns
    
    # Drop original date columns
    df_processed = df_processed.drop(columns=[col for col in date_columns if col in df_processed.columns])
    
    # Fill remaining NaN values with appropriate defaults
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'Rank':  # Don't fill target variable
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def train_investment_model(data_file='data.csv', model_file='investment_model.joblib'):
    """
    Train the investment scoring model
    """
    print("Loading training data...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} companies for training")
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df, is_training=True)
    print(f"After preprocessing: {len(df_processed)} companies")
    
    # Features and target
    X = df_processed.drop(['Rank'], axis=1)
    y = df_processed['Rank']
    
    print(f"Training features: {list(X.columns)}")
    print(f"Target range: {y.min():.1f} to {y.max():.1f}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Training XGBoost model...")
    # Train XGBoost regressor with some basic hyperparameters
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    print(f"\nSaving model to {model_file}...")
    dump(model, model_file)
    
    print("Training completed successfully!")
    return model, feature_importance

if __name__ == "__main__":
    # Train the model
    model, importance = train_investment_model()