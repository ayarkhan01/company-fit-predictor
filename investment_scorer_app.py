import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from predict_score import predict_single_company, predict_companies_from_file, load_model
import io

# Page config
st.set_page_config(
    page_title="Investment Fit Scorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .score-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .score-text {
        font-size: 4rem;
        font-weight: bold;
        color: white;
        margin: 0;
    }
    .score-label {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .recommendation {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .high-fit {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .medium-fit {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .low-fit {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def create_score_gauge(score):
    """Create a beautiful gauge chart for the investment score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Investment Fit Score", 'font': {'size': 24}},
        delta = {'reference': 5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 4], 'color': 'lightgray'},
                {'range': [4, 6], 'color': 'yellow'},
                {'range': [6, 8], 'color': 'orange'},
                {'range': [8, 10], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def get_recommendation(score):
    """Get investment recommendation based on score"""
    if score >= 8:
        return {
            'class': 'high-fit',
            'title': '🟢 STRONG INVESTMENT CANDIDATE',
            'text': 'This company shows excellent fit with your investment mandate. High priority for due diligence.',
            'action': 'Recommend proceeding with investment committee review'
        }
    elif score >= 6:
        return {
            'class': 'medium-fit',
            'title': '🟡 MODERATE FIT - INVESTIGATE FURTHER',
            'text': 'This company has potential but requires deeper analysis. Consider for pipeline.',
            'action': 'Schedule management presentation and detailed review'
        }
    else:
        return {
            'class': 'low-fit',
            'title': '🔴 POOR FIT',
            'text': 'This company does not align well with your investment criteria.',
            'action': 'Consider passing unless extraordinary circumstances'
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Investment Fit Scorer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Investment Decision Support System</p>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("📊 Navigation")
    mode = st.sidebar.selectbox(
        "Choose Scoring Mode",
        ["Single Company", "Multiple Companies"]
    )
    
    # Check if model exists
    try:
        model = load_model()
        st.sidebar.success("✅ Model loaded successfully")
    except FileNotFoundError:
        st.error("❌ Model not found! Please run `train_model.py` first.")
        st.stop()
    
    if mode == "Single Company":
        single_company_interface()
    elif mode == "Multiple Companies":
        batch_processing_interface()

def single_company_interface():
    st.header("🏢 Single Company")
    
    # Two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Company Metrics")
        
        employee_estimate = st.number_input(
            "Employee Count",
            min_value=1,
            max_value=100000,
            value=100,
            help="Total number of employees"
        )
        
        employees_networks = st.number_input(
            "Employees on Professional Networks",
            min_value=0,
            max_value=100000,
            value=50,
            help="LinkedIn, etc."
        )
        
        # Growth metrics
        st.write("**Growth Metrics (%)**")
        growth_monthly = st.number_input("Monthly Growth (%)", value=2.0, format="%.1f")
        growth_quarterly = st.number_input("Quarterly Growth (%)", value=6.0, format="%.1f")
        growth_6months = st.number_input("6-Month Growth (%)", value=12.0, format="%.1f")
        growth_annual = st.number_input("Annual Growth (%)", value=25.0, format="%.1f")
        
    with col2:
        st.subheader("🏛️ Company Details")
        
        company_name = st.text_input(
            "Company Name",
            value="Example Corp",
            help="Name of the company"
        )
        
        headquarters = st.text_input(
            "Headquarters",
            value="San Francisco, CA",
            help="City, State format"
        )
        
        year_founded = st.number_input(
            "Year Founded",
            min_value=1900,
            max_value=datetime.now().year,
            value=2015
        )
        
        business_model = st.selectbox(
            "Business Model",
            ["Software", "Software Enabled", "Tech", "Technology", "Hardware", "Services", "Other"]
        )
        
        # Funding details
        st.write("**Funding Information**")
        last_funding = st.number_input(
            "Last Funding Amount ($M)",
            min_value=0.0,
            value=10.0,
            format="%.1f"
        )
        
        total_funding = st.number_input(
            "Total Funding Amount ($M)",
            min_value=0.0,
            value=25.0,
            format="%.1f"
        )
        
        funding_rounds = st.number_input(
            "Total Funding Rounds",
            min_value=0,
            value=3
        )
        
        # Dates
        last_funding_date = st.date_input(
            "Last Funding Date",
            value=date(2023, 6, 15)
        )
    
    # Analyze button
    if st.button("🎯 Get Investment Score", type="primary", use_container_width=True):
        # Prepare company data
        company_data = {
            'Employee Estimate': employee_estimate,
            'Employees on Professional Networks': employees_networks,
            'Employee Growth (Monthly)': f"{growth_monthly}%",
            'Employee Growth (Quarterly)': f"{growth_quarterly}%",
            'Employee Growth (6 months)': f"{growth_6months}%",
            'Employee Growth (Annual)': f"{growth_annual}%",
            'Headquarters': headquarters,
            'Year Founded': year_founded,
            'Business Model': business_model,
            'Last Funding Amount': last_funding * 1000000,  # Convert to actual amount
            'Total Funding Amount': total_funding * 1000000,
            'Total Funding Rounds': funding_rounds,
            'Last Funding Date': last_funding_date.strftime('%Y-%m-%d')
        }
        
        # Predict score
        with st.spinner("Calculating score..."):
            try:
                score = predict_single_company(company_data)
                
                # Display results - just company name and score
                st.success("✅ Score calculated successfully!")
                
                # Simple result display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                        <h2 style="color: white; margin: 0;">{company_name}</h2>
                        <h1 style="color: white; font-size: 3rem; margin: 0.5rem 0;">{score}/10</h1>
                        <p style="color: rgba(255,255,255,0.9); margin: 0;">Investment Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error calculating score: {e}")

def batch_processing_interface():
    st.header("📁 Multiple Companies")
    
    st.info("Upload an Excel or CSV file with multiple companies to score them all at once.")
    
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['csv', 'xlsx'],
        help="File should contain company data with the same columns as single company analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} companies from {uploaded_file.name}")
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
            
            if st.button("🎯 Score All Companies", type="primary"):
                with st.spinner("Scoring companies..."):
                    # Save uploaded file temporarily
                    temp_file = f"temp_{uploaded_file.name}"
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process file
                    results = predict_companies_from_file(temp_file)
                    
                    # Display results - just company and score
                    st.subheader("📊 Company Scores")
                    
                    # Clean display - just company name and score
                    display_columns = []
                    if 'Name' in results.columns:
                        display_columns.append('Name')
                    elif 'Company' in results.columns:
                        display_columns.append('Company')
                    elif 'Company Name' in results.columns:
                        display_columns.append('Company Name')
                    
                    display_columns.append('Investment_Score')
                    
                    if display_columns and len(display_columns) == 2:
                        clean_df = results[display_columns].copy()
                        clean_df.columns = ['Company', 'Score']
                    else:
                        # Fallback if no company name column found
                        clean_df = results[['Investment_Score']].copy()
                        clean_df.columns = ['Score']
                        clean_df.insert(0, 'Company', [f'Company {i+1}' for i in range(len(clean_df))])
                    
                    # Style the dataframe with clean formatting
                    def style_score_row(row):
                        score = row['Score']
                        if score >= 8:
                            return ['background-color: #e8f5e8; color: #2d5a2d; font-weight: bold'] * len(row)
                        elif score >= 6:
                            return ['background-color: #fff8e1; color: #8b6914; font-weight: bold'] * len(row)
                        else:
                            return ['background-color: #fdeaea; color: #7a1a1a; font-weight: bold'] * len(row)
                    
                    styled_df = clean_df.style.apply(style_score_row, axis=1)
                    styled_df = styled_df.format({'Score': '{:.1f}'})
                    styled_df = styled_df.set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('color', '#262730'), ('font-weight', 'bold')]},
                        {'selector': 'td', 'props': [('text-align', 'left'), ('padding', '12px')]},
                        {'selector': 'tr:hover', 'props': [('background-color', '#f8f9fa')]}
                    ])
                    
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Download results
                    csv = clean_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"investment_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def portfolio_dashboard():
    st.header("📈 Portfolio Dashboard")
    st.info("This feature would show analytics across your investment pipeline and portfolio companies.")
    
    # Mock data for demonstration
    st.subheader("📊 Pipeline Overview")
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies Evaluated", "1,247", "+23 this week")
    
    with col2:
        st.metric("High Fit Companies", "156", "+5 this week")
    
    with col3:
        st.metric("Active Pipeline", "28", "+3 this week")
    
    with col4:
        st.metric("Investments Made", "12", "+1 this quarter")
    
    # Sample charts
    st.subheader("📈 Trends")
    
    # Mock data for trend chart
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
    scores = np.random.normal(6.5, 1.5, len(dates))
    trend_df = pd.DataFrame({'Date': dates, 'Average Score': scores})
    
    fig = px.line(trend_df, x='Date', y='Average Score', title='Average Investment Scores Over Time')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()