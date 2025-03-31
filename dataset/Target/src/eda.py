import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Optional

class DataAnalyzer:
    def __init__(self, data: pd.DataFrame):
        """Initialize DataAnalyzer with a dataset."""
        self.data = data
        
    def generate_demographics_summary(self) -> pd.DataFrame:
        """Generate summary statistics for demographic variables."""
        demographics = ['age', 'gender', 'height', 'weight', 'bmi']
        summary = {}
        
        for col in demographics:
            if col in self.data.columns:
                summary[col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'missing': self.data[col].isnull().sum()
                }
        
        return pd.DataFrame(summary).T
        
    def plot_distribution(self, feature: str, by: Optional[str] = None):
        """Plot distribution of a feature, optionally grouped by another variable."""
        plt.figure(figsize=(10, 6))
        
        if by is None:
            sns.histplot(data=self.data, x=feature, kde=True)
        else:
            sns.histplot(data=self.data, x=feature, hue=by, kde=True)
            
        plt.title(f'Distribution of {feature}')
        plt.show()
        
    def correlation_heatmap(self, features: List[str]):
        """Generate correlation heatmap for selected features."""
        corr = self.data[features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.show()
        
    def create_interactive_scatter(self, x: str, y: str, color: Optional[str] = None):
        """Create interactive scatter plot using plotly."""
        fig = px.scatter(self.data, x=x, y=y, color=color,
                        title=f'{y} vs {x}')
        return fig
