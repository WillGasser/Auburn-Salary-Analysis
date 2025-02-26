import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
import matplotlib.ticker as ticker
import os

# Set the style for the visualizations
plt.style.use('ggplot')
sns.set_palette("viridis")
sns.set_context("notebook", font_scale=1.2)

# Initialize text report
report_text = []

def add_to_report(text):
    """Add text to the report with a newline"""
    report_text.append(text)
    report_text.append("")  # Add a blank line

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
        'count': df['salary'].count(),
        'range': df['salary'].max() - df['salary'].min(),
        'percentile_25': df['salary'].quantile(0.25),
        'percentile_75': df['salary'].quantile(0.75),
        'iqr': df['salary'].quantile(0.75) - df['salary'].quantile(0.25),
        'skewness': df['salary'].skew(),
        'kurtosis': df['salary'].kurtosis(),
    }
    return stats

def get_stats_by_category(df, category):
    return df.groupby(category)['salary'].agg(['count', 'mean', 'median', 'min', 'max', 'std']).sort_values('mean', ascending=False)

# Text summary functions
def summarize_basic_stats(df, stats):
    """Generate text summary of basic statistics"""
    summary = []
    summary.append("# OVERALL SALARY STATISTICS")
    summary.append(f"Dataset contains {stats['count']} employees across {df['department'].nunique()} departments with {df['title'].nunique()} unique job titles.")
    
    summary.append("\n## Central Tendency and Dispersion")
    summary.append(f"Mean Salary: ${stats['mean']:,.2f}")
    summary.append(f"Median Salary: ${stats['median']:,.2f}")
    summary.append(f"Standard Deviation: ${stats['std']:,.2f}")
    summary.append(f"Minimum Salary: ${stats['min']:,.2f}")
    summary.append(f"Maximum Salary: ${stats['max']:,.2f}")
    summary.append(f"Salary Range: ${stats['range']:,.2f}")
    
    summary.append("\n## Quartile Analysis")
    summary.append(f"25th Percentile (Q1): ${stats['percentile_25']:,.2f}")
    summary.append(f"75th Percentile (Q3): ${stats['percentile_75']:,.2f}")
    summary.append(f"Interquartile Range (IQR): ${stats['iqr']:,.2f}")
    
    summary.append("\n## Distribution Shape")
    summary.append(f"Skewness: {stats['skewness']:.4f}")
    if stats['skewness'] > 1:
        skew_desc = "highly positively skewed (right-tailed)"
    elif stats['skewness'] > 0.5:
        skew_desc = "moderately positively skewed"
    elif stats['skewness'] > 0:
        skew_desc = "slightly positively skewed"
    elif stats['skewness'] < -1:
        skew_desc = "highly negatively skewed (left-tailed)"
    elif stats['skewness'] < -0.5:
        skew_desc = "moderately negatively skewed"
    else:
        skew_desc = "approximately symmetric"
    summary.append(f"The salary distribution is {skew_desc}.")
    
    summary.append(f"Kurtosis: {stats['kurtosis']:.4f}")
    if stats['kurtosis'] > 3:
        kurt_desc = "leptokurtic (heavy-tailed with more outliers)"
    elif stats['kurtosis'] < -3:
        kurt_desc = "platykurtic (light-tailed with fewer outliers)"
    else:
        kurt_desc = "mesokurtic (close to normal distribution in tail weight)"
    summary.append(f"The distribution is {kurt_desc}.")
    
    # Discuss the mean vs median
    ratio = stats['mean'] / stats['median']
    if ratio > 1.1:
        summary.append(f"\nThe mean salary (${stats['mean']:,.2f}) is substantially higher than the median (${stats['median']:,.2f}), with a ratio of {ratio:.2f}. This indicates the presence of high-value outliers pulling the mean upward.")
    elif ratio < 0.9:
        summary.append(f"\nThe mean salary (${stats['mean']:,.2f}) is lower than the median (${stats['median']:,.2f}), with a ratio of {ratio:.2f}. This suggests the presence of low-value outliers pulling the mean downward.")
    else:
        summary.append(f"\nThe mean salary (${stats['mean']:,.2f}) is relatively close to the median (${stats['median']:,.2f}), with a ratio of {ratio:.2f}, suggesting a somewhat symmetric distribution of salaries.")
    
    # Reference to the distribution figure
    summary.append("\n[Insert Figure: salary_distribution.png]")
    summary.append("Figure 1: Overall distribution of salaries showing the mean and median values.")
    
    return "\n".join(summary)

def summarize_department_analysis(df):
    """Generate text summary of department analysis"""
    dept_stats = get_stats_by_category(df, 'department')
    
    # Get top 10 departments by average salary
    top_depts = dept_stats.head(10)
    
    # Get bottom 10 departments by average salary
    bottom_depts = dept_stats.tail(10)
    
    # Get departments with highest variation (CV)
    dept_stats['cv'] = (dept_stats['std'] / dept_stats['mean']) * 100
    high_var_depts = dept_stats[dept_stats['count'] >= 3].sort_values('cv', ascending=False).head(10)
    
    # Calculate overall department statistics
    dept_size_mean = df.groupby('department').size().mean()
    dept_size_median = df.groupby('department').size().median()
    dept_size_max = df.groupby('department').size().max()
    largest_dept = df.groupby('department').size().idxmax()
    
    # Generate the summary text
    summary = []
    summary.append("# DEPARTMENT SALARY ANALYSIS")
    summary.append(f"Analysis of salaries across {df['department'].nunique()} different departments.")
    
    summary.append("\n## Department Size Statistics")
    summary.append(f"Average Department Size: {dept_size_mean:.1f} employees")
    summary.append(f"Median Department Size: {dept_size_median:.1f} employees")
    summary.append(f"Largest Department: {largest_dept} with {dept_size_max} employees")
    
    summary.append("\n## Highest Paying Departments")
    summary.append("Top 10 departments by average salary:")
    for i, (dept, row) in enumerate(top_depts.iterrows(), 1):
        summary.append(f"{i}. {dept}: ${row['mean']:,.2f} (n={row['count']}, range: ${row['min']:,.0f}-${row['max']:,.0f})")
    
    summary.append("\n## Lowest Paying Departments")
    summary.append("Bottom 10 departments by average salary:")
    for i, (dept, row) in enumerate(bottom_depts.iterrows(), 1):
        summary.append(f"{i}. {dept}: ${row['mean']:,.2f} (n={row['count']}, range: ${row['min']:,.0f}-${row['max']:,.0f})")
    
    summary.append("\n## Departments with Highest Salary Variation")
    summary.append("Top 10 departments by coefficient of variation (CV = std/mean):")
    for i, (dept, row) in enumerate(high_var_depts.iterrows(), 1):
        summary.append(f"{i}. {dept}: CV={row['cv']:.2f}% (n={row['count']}, mean=${row['mean']:,.2f}, std=${row['std']:,.2f})")
    
    # Correlation between department size and average salary
    dept_sizes = df.groupby('department').size()
    dept_avg_salaries = df.groupby('department')['salary'].mean()
    from scipy import stats as scipy_stats
    r, p = scipy_stats.pearsonr(dept_sizes, dept_avg_salaries)
    
    summary.append("\n## Relationship between Department Size and Salary")
    summary.append(f"Correlation coefficient: r = {r:.4f} (p-value: {p:.4f})")
    if p < 0.05:
        if r > 0:
            summary.append(f"There is a significant positive correlation between department size and average salary (r = {r:.4f}, p = {p:.4f}). Larger departments tend to have higher average salaries.")
        else:
            summary.append(f"There is a significant negative correlation between department size and average salary (r = {r:.4f}, p = {p:.4f}). Smaller departments tend to have higher average salaries.")
    else:
        summary.append(f"There is no significant correlation between department size and average salary (r = {r:.4f}, p = {p:.4f}).")
    
    # References to figures
    summary.append("\n[Insert Figure: department_boxplots.png]")
    summary.append("Figure 2: Box plots showing the distribution of salaries across the top departments.")
    
    summary.append("\n[Insert Figure: department_mean_salaries.png]")
    summary.append("Figure 3: Bar chart of mean salaries by department.")
    
    summary.append("\n[Insert Figure: department_variance.png]")
    summary.append("Figure 4: Departments ranked by salary variation (coefficient of variation).")
    
    summary.append("\n[Insert Figure: dept_size_salary_correlation.png]")
    summary.append("Figure 5: Scatter plot showing the relationship between department size and average salary.")
    
    return "\n".join(summary)

def summarize_title_analysis(df):
    """Generate text summary of job title analysis"""
    title_stats = get_stats_by_category(df, 'title')
    
    # Get top 10 titles by average salary
    top_titles = title_stats.head(10)
    
    # Get bottom 10 titles by average salary (with at least 5 employees)
    bottom_titles = title_stats[title_stats['count'] >= 5].tail(10)
    
    # Get titles with highest variation (CV)
    title_stats['cv'] = (title_stats['std'] / title_stats['mean']) * 100
    high_var_titles = title_stats[title_stats['count'] >= 5].sort_values('cv', ascending=False).head(10)
    
    # Calculate overall title statistics
    title_count_mean = df.groupby('title').size().mean()
    title_count_median = df.groupby('title').size().median()
    most_common_title = df['title'].value_counts().idxmax()
    most_common_count = df['title'].value_counts().max()
    
    # Generate the summary text
    summary = []
    summary.append("# JOB TITLE SALARY ANALYSIS")
    summary.append(f"Analysis of salaries across {df['title'].nunique()} different job titles.")
    
    summary.append("\n## Job Title Frequency Statistics")
    summary.append(f"Average number of employees per title: {title_count_mean:.1f}")
    summary.append(f"Median number of employees per title: {title_count_median:.1f}")
    summary.append(f"Most common job title: {most_common_title} with {most_common_count} employees")
    
    summary.append("\n## Highest Paying Job Titles")
    summary.append("Top 10 job titles by average salary:")
    for i, (title, row) in enumerate(top_titles.iterrows(), 1):
        summary.append(f"{i}. {title}: ${row['mean']:,.2f} (n={row['count']}, range: ${row['min']:,.0f}-${row['max']:,.0f})")
    
    summary.append("\n## Lowest Paying Job Titles")
    summary.append("Bottom 10 job titles by average salary (with at least 5 employees):")
    for i, (title, row) in enumerate(bottom_titles.iterrows(), 1):
        summary.append(f"{i}. {title}: ${row['mean']:,.2f} (n={row['count']}, range: ${row['min']:,.0f}-${row['max']:,.0f})")
    
    summary.append("\n## Job Titles with Highest Salary Variation")
    summary.append("Top 10 job titles by coefficient of variation (CV = std/mean, with at least 5 employees):")
    for i, (title, row) in enumerate(high_var_titles.iterrows(), 1):
        summary.append(f"{i}. {title}: CV={row['cv']:.2f}% (n={row['count']}, mean=${row['mean']:,.2f}, std=${row['std']:,.2f})")
    
    # References to figures
    summary.append("\n[Insert Figure: title_boxplots.png]")
    summary.append("Figure 6: Box plots showing the distribution of salaries across the top job titles.")
    
    summary.append("\n[Insert Figure: title_mean_salaries.png]")
    summary.append("Figure 7: Bar chart of mean salaries by job title.")
    
    summary.append("\n[Insert Figure: title_variance.png]")
    summary.append("Figure 8: Job titles ranked by salary variation (coefficient of variation).")
    
    return "\n".join(summary)

def summarize_title_department_relationship(df):
    """Generate text summary of the relationship between titles and departments"""
    # Create a pivot table of mean salaries
    top_departments = df['department'].value_counts().nlargest(10).index
    top_titles = df['title'].value_counts().nlargest(10).index
    
    df_filtered = df[df['department'].isin(top_departments) & df['title'].isin(top_titles)]
    pivot = df_filtered.pivot_table(
        values='salary', 
        index='department', 
        columns='title', 
        aggfunc='mean'
    )
    
    # Find the highest and lowest salary for each common title
    common_titles = df['title'].value_counts().head(5).index
    title_dept_variation = {}
    
    for title in common_titles:
        title_data = df[df['title'] == title]
        dept_counts = title_data['department'].value_counts()
        valid_depts = dept_counts[dept_counts >= 2].index
        
        if len(valid_depts) >= 2:
            dept_avg = title_data[title_data['department'].isin(valid_depts)].groupby('department')['salary'].mean()
            max_dept = dept_avg.idxmax()
            min_dept = dept_avg.idxmin()
            max_salary = dept_avg.max()
            min_salary = dept_avg.min()
            variation_pct = ((max_salary - min_salary) / min_salary) * 100
            
            title_dept_variation[title] = {
                'max_dept': max_dept,
                'min_dept': min_dept,
                'max_salary': max_salary,
                'min_salary': min_salary,
                'variation_pct': variation_pct,
                'dept_count': len(valid_depts)
            }
    
    # Generate the summary text
    summary = []
    summary.append("# TITLE-DEPARTMENT RELATIONSHIP ANALYSIS")
    summary.append("Analysis of how salaries for the same job title vary across different departments.")
    
    summary.append("\n## Cross-Department Title Comparison")
    summary.append("How the same job titles are compensated differently across departments:")
    
    for title, data in sorted(title_dept_variation.items(), key=lambda x: x[1]['variation_pct'], reverse=True):
        summary.append(f"\n### {title} (appears in {data['dept_count']} departments)")
        summary.append(f"Highest average salary: ${data['max_salary']:,.2f} in {data['max_dept']}")
        summary.append(f"Lowest average salary: ${data['min_salary']:,.2f} in {data['min_dept']}")
        summary.append(f"Salary variation: {data['variation_pct']:.1f}% difference between highest and lowest")
    
    # References to figures
    summary.append("\n[Insert Figure: title_dept_heatmap.png]")
    summary.append("Figure 9: Heatmap showing average salaries by department and title combinations.")
    
    summary.append("\n[Insert Figure: title_department_comparison.png]")
    summary.append("Figure 10: Comparison of how the same job titles are compensated across different departments.")
    
    return "\n".join(summary)

def summarize_outlier_analysis(df):
    """Generate text summary of salary outliers"""
    # Define function to detect outliers using IQR method
    def get_outliers(group):
        q1 = group['salary'].quantile(0.25)
        q3 = group['salary'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = group[(group['salary'] < lower_bound) | (group['salary'] > upper_bound)]
        return outliers
    
    # Get overall outliers
    q1 = df['salary'].quantile(0.25)
    q3 = df['salary'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    overall_outliers = df[(df['salary'] < lower_bound) | (df['salary'] > upper_bound)]
    high_outliers = df[df['salary'] > upper_bound]
    low_outliers = df[df['salary'] < lower_bound]
    
    # Get outliers by department and title
    dept_outliers = df.groupby('department', group_keys=False).apply(get_outliers).reset_index(drop=True)
    title_outliers = df.groupby('title', group_keys=False).apply(get_outliers).reset_index(drop=True)
    
    # Get departments and titles with most outliers
    dept_outlier_counts = dept_outliers['department'].value_counts().head(10)
    title_outlier_counts = title_outliers['title'].value_counts().head(10)
    
    # Generate the summary text
    summary = []
    summary.append("# SALARY OUTLIER ANALYSIS")
    summary.append("Analysis of salary outliers using the Interquartile Range (IQR) method.")
    
    summary.append("\n## Overall Outliers")
    summary.append(f"Total number of outliers: {len(overall_outliers)} ({len(overall_outliers)/len(df)*100:.1f}% of all employees)")
    summary.append(f"High outliers (above ${upper_bound:,.2f}): {len(high_outliers)} ({len(high_outliers)/len(df)*100:.1f}%)")
    summary.append(f"Low outliers (below ${lower_bound:,.2f}): {len(low_outliers)} ({len(low_outliers)/len(df)*100:.1f}%)")
    
    # Top 5 highest and lowest salaries
    top5_salaries = df.nlargest(5, 'salary')
    bottom5_salaries = df.nsmallest(5, 'salary')
    
    summary.append("\n## Extreme Values")
    summary.append("Top 5 highest salaries:")
    for i, (_, row) in enumerate(top5_salaries.iterrows(), 1):
        summary.append(f"{i}. ${row['salary']:,.2f} - {row['title']} in {row['department']}")
    
    summary.append("\nBottom 5 lowest salaries:")
    for i, (_, row) in enumerate(bottom5_salaries.iterrows(), 1):
        summary.append(f"{i}. ${row['salary']:,.2f} - {row['title']} in {row['department']}")
    
    summary.append("\n## Departments with Most Salary Outliers")
    for i, (dept, count) in enumerate(dept_outlier_counts.items(), 1):
        dept_total = len(df[df['department'] == dept])
        summary.append(f"{i}. {dept}: {count} outliers ({count/dept_total*100:.1f}% of department)")
    
    summary.append("\n## Job Titles with Most Salary Outliers")
    for i, (title, count) in enumerate(title_outlier_counts.items(), 1):
        title_total = len(df[df['title'] == title])
        summary.append(f"{i}. {title}: {count} outliers ({count/title_total*100:.1f}% of title)")
    
    # References to figures
    summary.append("\n[Insert Figure: outlier_analysis.png]")
    summary.append("Figure 11: Departments and job titles with the most salary outliers.")
    
    return "\n".join(summary)

def summarize_comprehensive_analysis(df, stats_dict):
    """Generate comprehensive summary and key insights"""
    # Find key metrics and insights
    highest_paid_dept = df.groupby('department')['salary'].mean().idxmax()
    highest_paid_dept_salary = df.groupby('department')['salary'].mean().max()
    
    lowest_paid_dept = df.groupby('department')['salary'].mean().idxmin()
    lowest_paid_dept_salary = df.groupby('department')['salary'].mean().min()
    
    highest_paid_title = df.groupby('title')['salary'].mean().idxmax()
    highest_paid_title_salary = df.groupby('title')['salary'].mean().max()
    
    # Filter titles with at least 5 employees first, then find the one with lowest mean salary
    title_counts = df['title'].value_counts()
    titles_with_5_plus = title_counts[title_counts >= 5].index
    df_filtered = df[df['title'].isin(titles_with_5_plus)]
    lowest_paid_title = df_filtered.groupby('title')['salary'].mean().idxmin()
    lowest_paid_title_salary = df_filtered.groupby('title')['salary'].mean().min()
    
    # Calculate coefficient of variation for the overall dataset
    overall_cv = (stats_dict['std'] / stats_dict['mean']) * 100
    
    # Generate the summary text
    summary = []
    summary.append("# COMPREHENSIVE ANALYSIS AND KEY INSIGHTS")
    summary.append("Summary of the most important findings from the salary analysis.")
    
    summary.append("\n## Salary Overview")
    summary.append(f"The analysis examined {stats_dict['count']} employees across {df['department'].nunique()} departments with {df['title'].nunique()} unique job titles.")
    summary.append(f"The average salary is ${stats_dict['mean']:,.2f}, with a median of ${stats_dict['median']:,.2f}.")
    summary.append(f"Salaries range from ${stats_dict['min']:,.2f} to ${stats_dict['max']:,.2f}, a difference of ${stats_dict['range']:,.2f}.")
    summary.append(f"The overall coefficient of variation is {overall_cv:.1f}%, indicating {'high' if overall_cv > 30 else 'moderate' if overall_cv > 15 else 'low'} salary dispersion.")
    
    summary.append("\n## Extreme Groups")
    summary.append(f"Highest paid department: {highest_paid_dept} (${highest_paid_dept_salary:,.2f} average)")
    summary.append(f"Lowest paid department: {lowest_paid_dept} (${lowest_paid_dept_salary:,.2f} average)")
    summary.append(f"Highest paid job title: {highest_paid_title} (${highest_paid_title_salary:,.2f} average)")
    summary.append(f"Lowest paid job title (with at least 5 employees): {lowest_paid_title} (${lowest_paid_title_salary:,.2f} average)")
    
    # Calculate the ratio between highest and lowest paid departments and titles
    dept_ratio = highest_paid_dept_salary / lowest_paid_dept_salary
    title_ratio = highest_paid_title_salary / lowest_paid_title_salary
    
    summary.append("\n## Inequality Measures")
    summary.append(f"The ratio between the highest and lowest paid departments is {dept_ratio:.2f}.")
    summary.append(f"The ratio between the highest and lowest paid job titles (with at least 5 employees) is {title_ratio:.2f}.")
    
    # Department size analysis
    dept_sizes = df.groupby('department').size()
    dept_avg_salaries = df.groupby('department')['salary'].mean()
    from scipy import stats as scipy_stats
    r, p = scipy_stats.pearsonr(dept_sizes, dept_avg_salaries)
    
    if p < 0.05:
        if r > 0:
            size_salary_insight = f"There is a significant positive correlation (r = {r:.2f}, p = {p:.4f}) between department size and average salary. Larger departments tend to have higher average salaries."
        else:
            size_salary_insight = f"There is a significant negative correlation (r = {r:.2f}, p = {p:.4f}) between department size and average salary. Smaller departments tend to have higher average salaries."
    else:
        size_salary_insight = f"There is no significant correlation (r = {r:.2f}, p = {p:.4f}) between department size and average salary."
    
    summary.append("\n## Key Insights")
    summary.append(size_salary_insight)
    
    # Title-department variation insight
    common_titles = df['title'].value_counts().head(5).index
    max_variation = 0
    max_var_title = ""
    
    for title in common_titles:
        title_data = df[df['title'] == title]
        dept_counts = title_data['department'].value_counts()
        valid_depts = dept_counts[dept_counts >= 2].index
        
        if len(valid_depts) >= 2:
            dept_avg = title_data[title_data['department'].isin(valid_depts)].groupby('department')['salary'].mean()
            variation_pct = ((dept_avg.max() - dept_avg.min()) / dept_avg.min()) * 100
            
            if variation_pct > max_variation:
                max_variation = variation_pct
                max_var_title = title
    
    if max_var_title:
        summary.append(f"The job title with the highest cross-department salary variation is '{max_var_title}', with a {max_variation:.1f}% difference between the highest and lowest paying departments.")
    
    # Distribution insight
    if stats_dict['skewness'] > 1:
        summary.append("The salary distribution is highly positively skewed, indicating a concentration of employees at lower salary levels with a smaller number of high earners pulling the mean upward.")
    
    # Reference to dashboard figure
    summary.append("\n[Insert Figure: salary_dashboard.png]")
    summary.append("Figure 12: Comprehensive dashboard providing an overview of key salary metrics and relationships.")
    
    return "\n".join(summary)

# Visualization functions (keeping them but focusing on text output)
def plot_salary_distribution(df, stats, output_path="salary_analysis_output/salary_distribution.png"):
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
    
    # Create a DataFrame for plotting
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

def plot_heatmap(df, output_path="salary_analysis_output/title_dept_heatmap.png"):
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

def plot_scatter_with_regression(df, output_path="salary_analysis_output/dept_size_salary_correlation.png"):
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
    from scipy import stats as scipy_stats
    r, p = scipy_stats.pearsonr(dept_sizes, dept_avg_salaries)
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

def create_dashboard(df, output_path="salary_analysis_output/salary_dashboard.png"):
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

def analyze_outliers(df, output_path="salary_analysis_output/outlier_analysis.png"):
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
    dept_outliers = df.groupby('department', group_keys=False).apply(get_outliers).reset_index(drop=True)
    title_outliers = df.groupby('title', group_keys=False).apply(get_outliers).reset_index(drop=True)
    
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

def compare_titles_across_departments(df, output_path="salary_analysis_output/title_department_comparison.png"):
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
    
    add_to_report(f"## SALARY ANALYSIS REPORT")
    add_to_report(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    add_to_report("")  # blank line
    
    # Calculate and add basic statistics to report
    stats = get_basic_stats(df)
    basic_stats_text = summarize_basic_stats(df, stats)
    add_to_report(basic_stats_text)
    
    # Department analysis
    dept_analysis_text = summarize_department_analysis(df)
    add_to_report(dept_analysis_text)
    
    # Title analysis
    title_analysis_text = summarize_title_analysis(df)
    add_to_report(title_analysis_text)
    
    # Title-department relationship analysis
    title_dept_text = summarize_title_department_relationship(df)
    add_to_report(title_dept_text)
    
    # Outlier analysis
    outlier_text = summarize_outlier_analysis(df)
    add_to_report(outlier_text)
    
    # Comprehensive analysis
    comprehensive_text = summarize_comprehensive_analysis(df, stats)
    add_to_report(comprehensive_text)
    
    # Create output directory if it doesn't exist
    if not os.path.exists('salary_analysis_output'):
        os.makedirs('salary_analysis_output')
    
    # Save the report to a text file
    report_path = "salary_analysis_output/salary_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_text))
    
    print(f"Text report saved to {report_path}")
    
    # Generate all visualizations
    print("Generating visualizations...")
    
    # Basic distribution
    plot_salary_distribution(df, stats, "salary_analysis_output/salary_distribution.png")
    
    # Department analysis
    plot_boxplots_by_category(df, 'department', top_n=15, output_path="salary_analysis_output/department_boxplots.png")
    plot_bar_chart_by_category(df, 'department', metric='mean', top_n=15, output_path="salary_analysis_output/department_mean_salaries.png")
    plot_bar_chart_by_category(df, 'department', metric='median', top_n=15, output_path="salary_analysis_output/department_median_salaries.png")
    
    # Title analysis
    plot_boxplots_by_category(df, 'title', top_n=15, output_path="salary_analysis_output/title_boxplots.png")
    plot_bar_chart_by_category(df, 'title', metric='mean', top_n=15, output_path="salary_analysis_output/title_mean_salaries.png")
    plot_bar_chart_by_category(df, 'title', metric='median', top_n=15, output_path="salary_analysis_output/title_median_salaries.png")
    
    # Cross-reference analysis
    plot_heatmap(df, output_path="salary_analysis_output/title_dept_heatmap.png")
    
    # Variance analysis
    plot_variance_analysis(df, 'department', output_path="salary_analysis_output/department_variance.png")
    plot_variance_analysis(df, 'title', output_path="salary_analysis_output/title_variance.png")
    
    # Correlation analysis
    plot_scatter_with_regression(df, output_path="salary_analysis_output/dept_size_salary_correlation.png")
    
    # Outlier analysis
    dept_outliers, title_outliers = analyze_outliers(df, output_path="salary_analysis_output/outlier_analysis.png")
    
    # Title comparison across departments
    compare_titles_across_departments(df, output_path="salary_analysis_output/title_department_comparison.png")
    
    # Create dashboard
    create_dashboard(df, output_path="salary_analysis_output/salary_dashboard.png")
    
    print(f"\nAnalysis complete! Text report and visualizations saved to the 'salary_analysis_output' directory.")
    
    # Print the report one more time in the console for easy access
    print("\n" + "="*80)
    print("REPORT TEXT (COPY FROM HERE):")
    print("="*80 + "\n")
    print("\n".join(report_text))
    print("\n" + "="*80)
    
    return df

if __name__ == "__main__":
    # Replace with the path to your JSON file
    file_path = "salary.json"
    df = run_analysis(file_path)