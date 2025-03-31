import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def analyze_bmi_category(bmi: float) -> str:
    """Categorize BMI according to WHO standards."""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def analyze_demographics(data: pd.DataFrame) -> Dict:
    """Analyze demographic data and provide clinical insights.
    
    Args:
        data (pd.DataFrame): DataFrame with columns [subject, gender, height, weight, age, bmi]
    
    Returns:
        Dict: Dictionary containing statistical analysis and clinical insights
    """
    stats = {}
    
    # Add BMI category
    data['bmi_category'] = data['bmi'].apply(analyze_bmi_category)
    
    # Basic statistics by gender
    stats['gender_stats'] = data.groupby('gender').agg({
        'age': ['count', 'mean', 'std'],
        'height': ['mean', 'std'],
        'weight': ['mean', 'std'],
        'bmi': ['mean', 'std']
    }).round(2)
    
    # BMI distribution
    stats['bmi_distribution'] = data['bmi_category'].value_counts()
    
    # Age groups
    data['age_group'] = pd.cut(data['age'], 
                              bins=[0, 30, 45, 60, 100],
                              labels=['18-30', '31-45', '46-60', '60+'])
    stats['age_distribution'] = data['age_group'].value_counts()
    
    return stats

def visualize_demographics(data: pd.DataFrame, output_dir: str = "plots"):
    """Create clinical visualizations for demographic data.
    
    Args:
        data (pd.DataFrame): DataFrame with demographic information
        output_dir (str): Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style to a default matplotlib style
    plt.style.use('default')
    # Use seaborn's color palette
    sns.set_palette("husl")
    
    # 1. BMI Distribution with WHO categories
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='bmi', bins=20)
    plt.axvline(x=18.5, color='r', linestyle='--', alpha=0.5, label='Underweight threshold')
    plt.axvline(x=25, color='y', linestyle='--', alpha=0.5, label='Overweight threshold')
    plt.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='Obese threshold')
    plt.title('BMI Distribution with WHO Categories')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'bmi_distribution.png'))
    plt.show()
    plt.close()
    
    # 2. Height vs Weight by Gender
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='height', y='weight', hue='gender', style='gender', s=100)
    plt.title('Height vs Weight by Gender')
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.savefig(os.path.join(output_dir, 'height_weight_scatter.png'))
    plt.show()
    plt.close()
    
    # 3. Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='age', bins=20)
    plt.title('Age Distribution')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    plt.show()
    plt.close()
    
    # 4. BMI vs Age by Gender
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='age', y='bmi', hue='gender', style='gender', s=100)
    plt.axhline(y=18.5, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=25, color='y', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    plt.title('BMI vs Age by Gender')
    plt.xlabel('Age (years)')
    plt.ylabel('BMI')
    plt.show()
    plt.savefig(os.path.join(output_dir, 'bmi_age_scatter.png'))
    plt.close()

def print_clinical_insights(stats: Dict):
    """Print clinical insights from demographic statistics."""
    print("\nClinical Statistics Summary")
    print("===========================")
    
    # Gender statistics
    print("\nStatistics by Gender:")
    print(stats['gender_stats'])
    
    print("\nBMI Distribution:")
    print(stats['bmi_distribution'])
    
    print("\nAge Distribution:")
    print(stats['age_distribution'])
    
    # Add clinical interpretations
    print("\nClinical Interpretations:")
    print("-------------------------")
    
    # BMI insights
    bmi_mean = stats['gender_stats']['bmi']['mean'].mean()
    print(f"- Average BMI: {bmi_mean:.1f}")
    if bmi_mean < 18.5:
        print("  - Population tends to be underweight")
    elif bmi_mean < 25:
        print("  - Population tends to be in normal weight range")
    elif bmi_mean < 30:
        print("  - Population tends to be overweight")
    else:
        print("  - Population tends to be obese")
    
    # Age insights
    age_mean = stats['gender_stats']['age']['mean'].mean()
    print(f"- Average Age: {age_mean:.1f} years")
    if age_mean < 30:
        print("  - Young adult population")
    elif age_mean < 50:
        print("  - Middle-aged adult population")
    else:
        print("  - Older adult population")
