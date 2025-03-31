import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clinical_stats import analyze_demographics, visualize_demographics, print_clinical_insights

# Example data
data = pd.DataFrame({
    'subject': ['s1'],
    'gender': ['female'],
    'height': [160],
    'weight': [50],
    'age': [25],
    'bmi': [19.53]  # Pre-calculated
})

# Analyze demographics
stats = analyze_demographics(data)

# Create visualizations
visualize_demographics(data)

# Print clinical insights
print_clinical_insights(stats)

# Additional clinical context for this specific case
print("\nSpecific Case Analysis:")
print("----------------------")
print("Subject s1:")
print(f"- BMI of 19.53 indicates normal weight status")
print("- Height of 160cm is within normal range for adult females")
print("- Weight of 50kg combined with height suggests balanced body composition")
print("- Age of 25 indicates young adult, typically associated with peak physical condition")
print("\nClinical Relevance:")
print("- Normal BMI suggests lower risk for weight-related health issues")
print("- Young age typically indicates good cardiovascular health")
print("- Female gender and age suggest regular hormonal patterns")
print("- Physical characteristics suggest normal metabolic function")
