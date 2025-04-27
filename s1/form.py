import os
from flask import Blueprint, render_template, current_app
import pandas as pd

# Create a blueprint for dataset analysis and preview
form_bp = Blueprint('form_bp', __name__, template_folder='templates')

@form_bp.route('/data-analysis')
def data_analysis():
    """
    Loads the dataset, performs advanced analysis, and renders the analysis page.
    
    Analysis includes:
      - A preview of the first 10 rows.
      - Descriptive statistics.
      - Missing values count per column.
      - A correlation matrix for numeric features.
      - Group analysis for 'Gender' and 'BMI_Post' (if available).
      - Frequency distribution of the target variable 'BMI_Post'.
    """
    # Construct the CSV file path relative to the app's root directory.
    csv_path = os.path.join(current_app.root_path, 'static', 'BMI_main.csv')
    
    try:
        df = pd.read_csv('static\BMI_main')
    except Exception as e:
        return render_template('error.html', message=f"Error loading dataset: {e}")
    
    # Generate a preview of the dataset (first 10 rows)
    preview = df.head(10).to_html(classes='table table-striped', index=False)
    
    # Compute descriptive statistics for numeric columns
    description = df.describe().to_html(classes='table table-bordered', float_format="{:.2f}".format)
    
    # Calculate missing values per column
    missing_values = df.isnull().sum().to_frame(name='Missing Values').to_html(classes='table table-hover')
    
    # Compute the correlation matrix for numeric columns
    correlation = df.corr().to_html(classes='table table-bordered', float_format="{:.2f}".format)
    
    # Basic insights about the dataset
    insights = (f"The dataset contains {df.shape[0]} records and {df.shape[1]} features. "
                "Review the tables below for detailed analysis.")
    
    # Advanced analysis: Group by 'Gender' for the 'BMI_Post' column if available
    if 'Gender' in df.columns and 'BMI_Post' in df.columns:
        group_analysis = (df.groupby('Gender')['BMI_Post']
                          .agg(['mean', 'median', 'std'])
                          .to_html(classes='table table-bordered', float_format="{:.2f}".format))
    else:
        group_analysis = "Gender-based analysis is not available."
    
    # Frequency distribution of the target variable 'BMI_Post'
    if 'BMI_Post' in df.columns:
        # Create bins based on the min, quartiles, and max
        bins = [df['BMI_Post'].min()] + list(df['BMI_Post'].quantile([0.25, 0.5, 0.75]).values) + [df['BMI_Post'].max()]
        freq_distribution = (pd.cut(df['BMI_Post'], bins=bins, include_lowest=True)
                             .value_counts()
                             .sort_index()
                             .to_frame(name='Frequency')
                             .to_html(classes='table table-bordered'))
    else:
        freq_distribution = "Frequency distribution for BMI_Post is not available."
    
    return render_template('data_analysis.html',
                           preview=preview,
                           description=description,
                           missing_values=missing_values,
                           correlation=correlation,
                           insights=insights,
                           group_analysis=group_analysis,
                           freq_distribution=freq_distribution)
