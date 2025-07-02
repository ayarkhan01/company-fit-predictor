import pandas as pd
import numpy as np
from datetime import datetime
from joblib import load
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, is_training=False):
    """
    Preprocess the data for both training and prediction
    (Same function as in train_model.py)
    """
    # Select relevant columns for prediction
    columns = ['Employee Estimate', 'Employees on Professional Networks', 'Employee Growth (Monthly)', 
               'Employee Growth (Quarterly)', 'Employee Growth (6 months)', 'Employee Growth (Annual)', 
               'Headquarters', 'Year Founded', 'Last Funding Amount', 'Last Funding Date', 
               'Total Funding Amount', 'Total Funding Rounds', 'Business Model', 'Last Touch Date', 
               'Last Pipeline Decline Date']
    
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
    
    # Clean data - drop original columns that were transformed
    columns_to_drop = ['Year Founded', 'Headquarters', 'Business Model']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
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
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def load_model(model_file='investment_model.joblib'):
    """
    Load the trained model
    """
    try:
        model = load(model_file)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{model_file}' not found. Please train the model first using train_model.py")

def predict_single_company(company_data, model_file='investment_model.joblib'):
    """
    Predict investment score for a single company
    
    Args:
        company_data: dict with company information
        model_file: path to saved model
    
    Returns:
        float: Investment fit score (1-10)
    """
    # Load model
    model = load_model(model_file)
    
    # Convert dict to DataFrame
    df = pd.DataFrame([company_data])
    
    # Preprocess
    df_processed = preprocess_data(df, is_training=False)
    
    # Predict (model returns 0-9, so add 1 to get 1-10 scale)
    prediction = model.predict(df_processed)[0] + 1
    
    # Cap to 1-10 range and round to 1 decimal
    score = max(1.0, min(10.0, round(prediction, 1)))
    
    return score

def predict_companies_from_file(file_path, model_file='investment_model.joblib', output_file=None):
    """
    Predict investment scores for multiple companies from Excel/CSV file
    
    Args:
        file_path: path to Excel or CSV file
        model_file: path to saved model
        output_file: optional path to save results
    
    Returns:
        DataFrame: Companies with scores, sorted by score (highest first)
    """
    # Load model
    model = load_model(model_file)
    
    # Load data
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} companies from {file_path}")
    
    # Keep original data for results
    original_df = df.copy()
    
    # Preprocess for prediction
    df_processed = preprocess_data(df, is_training=False)
    
    # Predict (model returns 0-9, so add 1 to get 1-10 scale)
    predictions = model.predict(df_processed) + 1
    
    # Cap to 1-10 range and round to 1 decimal
    scores = [max(1.0, min(10.0, round(pred, 1))) for pred in predictions]
    
    # Add scores to original data
    result_df = original_df.copy()
    result_df['Investment_Score'] = scores
    
    # Sort by score (highest first)
    result_df = result_df.sort_values('Investment_Score', ascending=False).reset_index(drop=True)
    
    # Save to file if requested
    if output_file:
        if output_file.endswith('.xlsx'):
            result_df.to_excel(output_file, index=False)
        else:
            result_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return result_df

def predict_companies_interactive():
    """
    Interactive mode for predicting company scores
    """
    print("Investment Fit Scoring Tool")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Score a single company (manual input)")
        print("2. Score companies from file")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            print("\nEnter company information (press Enter to skip optional fields):")
            company_data = {}
            
            # Essential fields
            fields = [
                ('Employee Estimate', int, False),
                ('Employees on Professional Networks', int, True),
                ('Employee Growth (Monthly)', str, True),
                ('Employee Growth (Quarterly)', str, True),
                ('Employee Growth (6 months)', str, True),
                ('Employee Growth (Annual)', str, True),
                ('Headquarters', str, True),
                ('Year Founded', int, True),
                ('Last Funding Amount', float, True),
                ('Total Funding Amount', float, True),
                ('Total Funding Rounds', int, True),
                ('Business Model', str, True)
            ]
            
            for field_name, field_type, optional in fields:
                while True:
                    value = input(f"{field_name}{'(optional)' if optional else ''}: ").strip()
                    if not value and optional:
                        break
                    elif not value and not optional:
                        print("This field is required.")
                        continue
                    
                    try:
                        if field_type == int:
                            company_data[field_name] = int(value)
                        elif field_type == float:
                            company_data[field_name] = float(value)
                        else:
                            company_data[field_name] = value
                        break
                    except ValueError:
                        print(f"Please enter a valid {field_type.__name__}")
            
            try:
                score = predict_single_company(company_data)
                print(f"\n🎯 Investment Fit Score: {score}/10")
                if score >= 8:
                    print("💰 HIGH FIT - Strong investment candidate")
                elif score >= 6:
                    print("🔍 MEDIUM FIT - Worth investigating further")
                else:
                    print("❌ LOW FIT - Likely pass")
            except Exception as e:
                print(f"Error predicting score: {e}")
        
        elif choice == '2':
            file_path = input("Enter path to Excel/CSV file: ").strip()
            output_path = input("Enter output file path (optional, press Enter to skip): ").strip()
            
            try:
                results = predict_companies_from_file(
                    file_path, 
                    output_file=output_path if output_path else None
                )
                
                print(f"\n📊 Results Summary:")
                print(f"Total companies: {len(results)}")
                print(f"High fit (8-10): {len(results[results['Investment_Score'] >= 8])}")
                print(f"Medium fit (6-7.9): {len(results[(results['Investment_Score'] >= 6) & (results['Investment_Score'] < 8)])}")
                print(f"Low fit (1-5.9): {len(results[results['Investment_Score'] < 6])}")
                
                print(f"\n🏆 Top 10 Companies:")
                if 'Name' in results.columns:
                    top_companies = results[['Name', 'Investment_Score']].head(10)
                elif 'Company' in results.columns:
                    top_companies = results[['Company', 'Investment_Score']].head(10)
                else:
                    top_companies = results[['Investment_Score']].head(10)
                
                print(top_companies.to_string(index=False))
                
            except FileNotFoundError:
                print("File not found. Please check the file path.")
            except Exception as e:
                print(f"Error processing file: {e}")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    # Run interactive mode
    predict_companies_interactive()