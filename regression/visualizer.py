import matplotlib.pyplot as plt
import seaborn as sns
from config import COLORS, FIGURE_DPI, PLOT_STYLE

class InsuranceVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style(PLOT_STYLE)

    def save_plot(self, name):
        plt.savefig(self.output_dir / name, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

    def plot_correlations(self, corr_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.2f')
        plt.title('Corelation matric')
        self.save_plot('correlation_matrix.png')

    def plot_distributions(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.boxplot(data=df, x='smoker', y='charges', hue='smoker', 
                    palette=COLORS, ax=axes[0], legend=False)
        
        sns.violinplot(data=df, x='smoker', y='charges', hue='smoker', 
                       palette=COLORS, ax=axes[1], legend=False)
        
        axes[0].set_title('Charges (Boxplot)')
        axes[1].set_title('Density distribution (Violin Plot)')
        self.save_plot('distributions.png')

    def plot_regression_residuals(self, model):
        plt.figure(figsize=(10, 6))
        sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, 
                      line_kws={'color': 'red', 'lw': 2})
        plt.title('Model residual analysis (Residuals Plot)')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        self.save_plot('regression_residuals.png')