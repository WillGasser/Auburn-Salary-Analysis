import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
import matplotlib.ticker as ticker

# Set the style for the visualizations
plt.style.use('ggplot')
sns.set_palette("viridis")
sns.set_context("notebook", font_scale=1.2)

# Load the data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Basic statistical analysis functions
def get_basic_stats(df):
    stats = {
        'mean': df['salary'].mean(),
        'median': df['salary'].median(),
        'min': df['salary'].min(),
        'max': df['salary'].max(),
        'std': df['salary'].std(),
        'count': df['salary'].count()
    }
    return stats

def get_stats_by_category(df, category):
    return df.groupby(category)['salary'].agg(['count', 'mean', 'median', 'min', 'max', 'std']).sort_values('mean', ascending=False)

# Visualization functions
def plot_salary_distribution(df, stats, output_path="salary_distribution.png"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram with KDE
    sns.histplot(df['salary'], kde=True, ax=ax, bins=30, color='skyblue')
    
    # Add vertical lines for key statistics
    plt.axvline(stats['mean'], color='r', linestyle='--', label=f'Mean: ${stats["mean"]:,.0f}')
    plt.axvline(stats['median'], color='g', linestyle='-', label=f'Median: ${stats["median"]:,.0f}')
    
    # Add annotations
    plt.text(stats['max']*0.98, plt.gca().get_ylim()[1]*0.9, f'Max: ${stats["max"]:,.0f}', 
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(stats['min']*1.02, plt.gca().get_ylim()[1]*0.9, f'Min: ${stats["min"]:,.0f}', 
             horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    
    # Format the x-axis with dollar signs
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax.xaxis.set_major_formatter(formatter)
    
    plt.title('Salary Distribution Overview', fontsize=16)
    plt.xlabel('Salary ($)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_boxplots_by_category(df, category, top_n=10, output_path=None):
    # Get the top N categories by count
    top_categories = df[category].value_counts().nlargest(top_n).index
    df_filtered = df[df[category].isin(top_categories)]
    
    # Calculate average salary for each category for sorting
    avg_salaries = df_filtered.groupby(category)['salary'].mean().sort_values(ascending=False)
    category_order = avg_salaries.index
    
    plt.figure(figsize=(14, 10))
    ax = sns.boxplot(x=category, y='salary', data=df_filtered, order=category_order)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Format y-axis with dollar signs
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.title(f'Salary Distribution by {category.capitalize()}', fontsize=16)
    plt.xlabel(category.capitalize(), fontsize=14)
    plt.ylabel('Salary ($)', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.close()

def plot_bar_chart_by_category(df, category, metric='mean', top_n=10, output_path=None):
    # Calculate the metric (e.g., mean, median) for each category
    if metric == 'mean':
        data = df.groupby(category)['salary'].mean().sort_values(ascending=False).head(top_n)
    elif metric == 'median':
        data = df.groupby(category)['salary'].median().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(14, 8))
    
    # Create a DataFrame for barplot to avoid the deprecation warning
    plot_df = pd.DataFrame({
        'category': data.index,
        'value': data.values
    })
    
    # Use hue parameter with legend=False to avoid the warning
    ax = sns.barplot(x='category', y='value', data=plot_df, hue='category', palette='viridis', legend=False)
    
    # Add value labels on top of each bar
    for i, v in enumerate(data.values):
        ax.text(i, v + 1000, f'${v:,.0f}', ha='center', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Format y-axis with dollar signs
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.title(f'{metric.capitalize()} Salary by {category.capitalize()}', fontsize=16)
    plt.xlabel(category.capitalize(), fontsize=14)
    plt.ylabel(f'{metric.capitalize()} Salary ($)', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.close()

def plot_heatmap(df, output_path="title_dept_heatmap.png"):
    # Get the most common departments and titles (top 10 each)
    top_departments = df['department'].value_counts().nlargest(10).index
    top_titles = df['title'].value_counts().nlargest(10).index
    
    # Filter data to include only the top departments and titles
    df_filtered = df[df['department'].isin(top_departments) & df['title'].isin(top_titles)]
    
    # Create a pivot table with average salaries
    pivot = df_filtered.pivot_table(
        values='salary', 
        index='department', 
        columns='title', 
        aggfunc='mean'
    )
    
    # Create a function to format the annotation text
    def fmt_salary(val):
        return f"${val:,.0f}" if not np.isnan(val) else ""
    
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5, 
                    cbar_kws={'label': 'Average Salary ($)'})
    
    # Update the annotations to include dollar signs
    for text in ax.texts:
        try:
            value = float(text.get_text())
            text.set_text(f"${value:,.0f}")
        except ValueError:
            pass
    
    plt.title('Average Salary by Department and Title', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_variance_analysis(df, category, output_path=None):
    # Calculate coefficient of variation (CV) for each category
    # CV = std / mean (standardized measure of dispersion)
    stats_df = get_stats_by_category(df, category)
    stats_df['cv'] = (stats_df['std'] / stats_df['mean']) * 100
    
    # Filter to categories with at least 3 employees
    stats_df = stats_df[stats_df['count'] >= 3].sort_values('cv', ascending=False).head(15)
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame({
        'category': stats_df.index,
        'cv': stats_df['cv']
    })
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='category', y='cv', data=plot_df, hue='category', palette='plasma', legend=False)
    
    # Add value labels on top of each bar
    for i, v in enumerate(stats_df['cv']):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.title(f'Salary Variation by {category.capitalize()} (Coefficient of Variation)', fontsize=16)
    plt.xlabel(category.capitalize(), fontsize=14)
    plt.ylabel('Coefficient of Variation (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.close()

def plot_scatter_with_regression(df, output_path="scatter_regression.png"):
    # Create a new column with department size
    dept_sizes = df.groupby('department').size()
    df['dept_size'] = df['department'].map(dept_sizes)
    
    # Create a new column with average department salary
    dept_avg_salaries = df.groupby('department')['salary'].mean()
    df['dept_avg_salary'] = df['department'].map(dept_avg_salaries)
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(x='dept_size', y='dept_avg_salary', data=df.drop_duplicates('department'), 
                      alpha=0.7, s=100)
    
    # Add regression line
    sns.regplot(x='dept_size', y='dept_avg_salary', data=df.drop_duplicates('department'), 
               scatter=False, ax=ax, color='red', line_kws={"linestyle": "--"})
    
    # Calculate correlation coefficient and p-value
    r, p = stats.pearsonr(dept_sizes, dept_avg_salaries)
    plt.title(f'Relationship Between Department Size and Average Salary\nCorrelation: {r:.2f} (p-value: {p:.4f})', fontsize=16)
    
    # Format y-axis with dollar signs
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    # Add department labels to points
    for i, row in df.drop_duplicates('department').iterrows():
        plt.text(row['dept_size'] + 0.2, row['dept_avg_salary'], 
                 row['department'], fontsize=8, alpha=0.7)
    
    plt.xlabel('Department Size (Number of Employees)', fontsize=14)
    plt.ylabel('Average Department Salary ($)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_dashboard(df, output_path="salary_dashboard.png"):
    # Create overall basic stats
    stats = get_basic_stats(df)
    
    # Set up the figure
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 3, figure=fig)
    
    # Plot 1: Overall salary distribution
    ax1 = fig.add_subplot(gs[0, :])
    sns.histplot(df['salary'], kde=True, ax=ax1, bins=30, color='skyblue')
    ax1.axvline(stats['mean'], color='r', linestyle='--', label=f'Mean: ${stats["mean"]:,.0f}')
    ax1.axvline(stats['median'], color='g', linestyle='-', label=f'Median: ${stats["median"]:,.0f}')
    ax1.set_title('Overall Salary Distribution', fontsize=16)
    ax1.set_xlabel('Salary ($)', fontsize=14)
    ax1.legend()
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax1.xaxis.set_major_formatter(formatter)
    
    # Plot 2: Top 5 highest paid departments
    ax2 = fig.add_subplot(gs[1, 0])
    dept_means = df.groupby('department')['salary'].mean().sort_values(ascending=False).head(5)
    dept_df = pd.DataFrame({
        'department': dept_means.index,
        'mean_salary': dept_means.values
    })
    sns.barplot(x='mean_salary', y='department', data=dept_df, ax=ax2, palette='viridis')
    ax2.set_title('Top 5 Highest Paid Departments', fontsize=16)
    ax2.set_xlabel('Average Salary ($)', fontsize=14)
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax2.xaxis.set_major_formatter(formatter)
    
    # Plot 3: Top 5 highest paid titles
    ax3 = fig.add_subplot(gs[1, 1])
    title_means = df.groupby('title')['salary'].mean().sort_values(ascending=False).head(5)
    title_df = pd.DataFrame({
        'title': title_means.index,
        'mean_salary': title_means.values
    })
    sns.barplot(x='mean_salary', y='title', data=title_df, ax=ax3, palette='plasma')
    ax3.set_title('Top 5 Highest Paid Titles', fontsize=16)
    ax3.set_xlabel('Average Salary ($)', fontsize=14)
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax3.xaxis.set_major_formatter(formatter)
    
    # Plot 4: Salary boxplot by title (top 5)
    ax4 = fig.add_subplot(gs[1, 2])
    top_titles = df['title'].value_counts().nlargest(5).index
    df_filtered = df[df['title'].isin(top_titles)]
    sns.boxplot(x='salary', y='title', data=df_filtered, ax=ax4, palette='viridis')
    ax4.set_title('Salary Distribution by Top 5 Titles', fontsize=16)
    ax4.set_xlabel('Salary ($)', fontsize=14)
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax4.xaxis.set_major_formatter(formatter)
    
    # Plot 5: Scatter plot of department size vs. average salary
    ax5 = fig.add_subplot(gs[2, 0:2])
    dept_sizes = df.groupby('department').size()
    dept_avg_salaries = df.groupby('department')['salary'].mean()
    
    # Create temporary dataframe for plotting
    temp_df = pd.DataFrame({
        'department': dept_sizes.index,
        'size': dept_sizes.values,
        'avg_salary': dept_avg_salaries.values
    })
    
    sns.scatterplot(x='size', y='avg_salary', data=temp_df, ax=ax5, alpha=0.7, s=100)
    ax5.set_title('Department Size vs. Average Salary', fontsize=16)
    ax5.set_xlabel('Department Size (Number of Employees)', fontsize=14)
    ax5.set_ylabel('Average Salary ($)', fontsize=14)
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax5.yaxis.set_major_formatter(formatter)
    
    # Plot 6: Salary variation coefficient by title
    ax6 = fig.add_subplot(gs[2, 2])
    title_stats = get_stats_by_category(df, 'title')
    title_stats['cv'] = (title_stats['std'] / title_stats['mean']) * 100
    title_stats = title_stats[title_stats['count'] >= 3].sort_values('cv', ascending=False).head(5)
    
    # Create dataframe for plotting
    cv_df = pd.DataFrame({
        'title': title_stats.index,
        'cv': title_stats['cv']
    })
    sns.barplot(x='cv', y='title', data=cv_df, ax=ax6, palette='magma')
    ax6.set_title('Salary Variation by Title', fontsize=16)
    ax6.set_xlabel('Coefficient of Variation (%)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def analyze_outliers(df, output_path="outlier_analysis.png"):
    # Define a function to detect outliers using IQR method
    def get_outliers(group):
        q1 = group['salary'].quantile(0.25)
        q3 = group['salary'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = group[(group['salary'] < lower_bound) | (group['salary'] > upper_bound)]
        return outliers
    
    # Get outliers by department and title
    dept_outliers = df.groupby('department').apply(get_outliers).reset_index(drop=True)
    title_outliers = df.groupby('title').apply(get_outliers).reset_index(drop=True)
    
    # Create a figure to display outlier information
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Plot 1: Number of outliers by department
    dept_outlier_counts = dept_outliers['department'].value_counts().nlargest(10)
    dept_df = pd.DataFrame({
        'department': dept_outlier_counts.index,
        'count': dept_outlier_counts.values
    })
    sns.barplot(x='count', y='department', data=dept_df, ax=ax1, palette='viridis')
    ax1.set_title('Departments with Most Salary Outliers', fontsize=16)
    ax1.set_xlabel('Number of Outliers', fontsize=14)
    
    # Plot 2: Number of outliers by title
    title_outlier_counts = title_outliers['title'].value_counts().nlargest(10)
    title_df = pd.DataFrame({
        'title': title_outlier_counts.index,
        'count': title_outlier_counts.values
    })
    sns.barplot(x='count', y='title', data=title_df, ax=ax2, palette='plasma')
    ax2.set_title('Titles with Most Salary Outliers', fontsize=16)
    ax2.set_xlabel('Number of Outliers', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Return the outliers dataframes for further analysis
    return dept_outliers, title_outliers

def compare_titles_across_departments(df, output_path="title_department_comparison.png"):
    # Get the 5 most common titles that appear in multiple departments
    common_titles = df['title'].value_counts().head(5).index
    
    # Filter the dataframe to include only these titles
    df_filtered = df[df['title'].isin(common_titles)]
    
    # For each title, get the average salary by department
    plt.figure(figsize=(16, 12))
    
    for i, title in enumerate(common_titles):
        plt.subplot(len(common_titles), 1, i+1)
        
        title_data = df[df['title'] == title]
        # Only include departments with at least 2 employees with this title
        dept_counts = title_data['department'].value_counts()
        valid_depts = dept_counts[dept_counts >= 2].index
        title_data = title_data[title_data['department'].isin(valid_depts)]
        
        if len(title_data) > 0:
            # Calculate average salary by department for this title
            dept_avg = title_data.groupby('department')['salary'].mean().sort_values(ascending=False).head(10)
            
            # Create a DataFrame for plotting
            plot_df = pd.DataFrame({
                'department': dept_avg.index,
                'avg_salary': dept_avg.values
            })
            
            ax = sns.barplot(x='department', y='avg_salary', data=plot_df, hue='department', legend=False)
            
            # Format y-axis with dollar signs
            formatter = ticker.StrMethodFormatter('${x:,.0f}')
            ax.yaxis.set_major_formatter(formatter)
            
            plt.title(f'Average Salary for {title} Across Departments', fontsize=14)
            plt.ylabel('Average Salary ($)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add counts on top of bars
            for j, v in enumerate(dept_avg.values):
                dept = dept_avg.index[j]
                count = len(title_data[title_data['department'] == dept])
                ax.text(j, v + 1000, f'n={count}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_analysis(file_path):
    # Load data
    df = load_data(file_path)
    
    # Print basic information about the dataset
    print(f"Dataset loaded with {len(df)} employees")
    print(f"Number of unique departments: {df['department'].nunique()}")
    print(f"Number of unique job titles: {df['title'].nunique()}")
    
    # Calculate and print basic statistics
    stats = get_basic_stats(df)
    print("\nOverall Salary Statistics:")
    for key, value in stats.items():
        if key in ['mean', 'median', 'min', 'max', 'std']:
            print(f"{key.capitalize()}: ${value:,.2f}")
        else:
            print(f"{key.capitalize()}: {value}")
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('fixed_analysis_output'):
        os.makedirs('fixed_analysis_output')
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    # Basic distribution
    plot_salary_distribution(df, stats, "fixed_analysis_output/salary_distribution.png")
    
    # Department analysis
    plot_boxplots_by_category(df, 'department', top_n=15, output_path="fixed_analysis_output/department_boxplots.png")
    plot_bar_chart_by_category(df, 'department', metric='mean', top_n=15, output_path="fixed_analysis_output/department_mean_salaries.png")
    plot_bar_chart_by_category(df, 'department', metric='median', top_n=15, output_path="fixed_analysis_output/department_median_salaries.png")
    
    # Title analysis
    plot_boxplots_by_category(df, 'title', top_n=15, output_path="fixed_analysis_output/title_boxplots.png")
    plot_bar_chart_by_category(df, 'title', metric='mean', top_n=15, output_path="fixed_analysis_output/title_mean_salaries.png")
    plot_bar_chart_by_category(df, 'title', metric='median', top_n=15, output_path="fixed_analysis_output/title_median_salaries.png")
    
    # Cross-reference analysis
    plot_heatmap(df, output_path="fixed_analysis_output/title_dept_heatmap.png")
    
    # Variance analysis
    plot_variance_analysis(df, 'department', output_path="fixed_analysis_output/department_variance.png")
    plot_variance_analysis(df, 'title', output_path="fixed_analysis_output/title_variance.png")
    
    # Correlation analysis
    plot_scatter_with_regression(df, output_path="fixed_analysis_output/dept_size_salary_correlation.png")
    
    # Outlier analysis
    dept_outliers, title_outliers = analyze_outliers(df, output_path="fixed_analysis_output/outlier_analysis.png")
    
    # Title comparison across departments
    compare_titles_across_departments(df, output_path="fixed_analysis_output/title_department_comparison.png")
    
    # Create dashboard
    create_dashboard(df, output_path="fixed_analysis_output/salary_dashboard.png")
    
    print("\nAnalysis complete! All results saved to the 'fixed_analysis_output' directory.")
    print("Key insights summary:")
    
    # Generate some key insights
    highest_paid_dept = df.groupby('department')['salary'].mean().sort_values(ascending=False).head(1)
    highest_paid_title = df.groupby('title')['salary'].mean().sort_values(ascending=False).head(1)
    most_variable_dept = df.groupby('department').agg({'salary': ['std', 'mean']})
    most_variable_dept['cv'] = most_variable_dept[('salary', 'std')] / most_variable_dept[('salary', 'mean')] * 100
    most_variable_dept = most_variable_dept.sort_values(('cv'), ascending=False).head(1)
    
    print(f"- Highest paid department: {highest_paid_dept.index[0]} (${highest_paid_dept.values[0]:,.2f} avg)")
    print(f"- Highest paid title: {highest_paid_title.index[0]} (${highest_paid_title.values[0]:,.2f} avg)")
    print(f"- Department with most salary variation: {most_variable_dept.index[0]} ({most_variable_dept[('cv')].values[0]:.1f}% CV)")
    
    # Return the dataframe for any additional analysis
    return df

if __name__ == "__main__":
    # Replace with the path to your JSON file
    file_path = "salary.json"
    df = run_analysis(file_path)