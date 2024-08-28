---
layout: post
title: "Human Resources Data Analysis"
image: "/assets/projects/Human Resources Data Analysis.webp"
date: 2023-11-27
excerpt_separator: <!--more-->
tags: [Data Science,data Analysis, Python]
mathjax: "true"
---
In this project, we embark on a journey of HR Analytics to analyze and visualize our company's extensive dataset.
<!--more-->

# Human Resources Data Analysis
The multifaceted role of Human Resources professionals transcends the perceived simplicity of the job. HR experts engage in a spectrum of workplace activities, from spearheading recruitment efforts and resolving employee issues, to nurturing a positive work environment, and critically evaluating performance and efficiency.

Among their various duties, one of the most intricate tasks HR professionals face is the assessment of employee performance and efficiency. The complexity of this responsibility scales with the size of the employee base, posing a significant challenge to HR's capability to manage effectively. In response to this challenge, we have initiated an Exploratory Data Analysis (EDA) project.

Within the scope of this project, we aim to conduct thorough analyses and create insightful visualizations using the Employee Engagement Dataset. Our objective is to derive valuable insights that can inform and refine HR practices. The project will dissect data on performance scores, scrutinize the link between performance and remuneration, investigate factors influencing employee attrition, and define the contours of employee satisfaction across different departments.

This analysis endeavors to arm HR professionals with robust data-driven insights, bolstering their ability to foster workforce development and support. By leveraging this strategic approach, we can enhance the impact and operational efficiency of HR functions.

This work is conducted under the dataset license "CC-BY-NC-ND: This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License." The license governs the use of the dataset, ensuring that it is utilized in a manner that aligns with the stipulated non-commercial and no-derivatives terms.

## Import Necessary Libraries

First, let's import the libraries that we will need.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
```

## Data Loading

Load the dataset into a Pandas DataFrame.


```python
dataset = pd.read_csv('HRDataset_v14.csv')
```


```python
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 311 entries, 0 to 310
    Data columns (total 38 columns):
     #   Column                      Non-Null Count  Dtype         
    ---  ------                      --------------  -----         
     0   Employee_Name               311 non-null    object        
     1   EmpID                       311 non-null    int64         
     2   MarriedID                   311 non-null    int64         
     3   MaritalStatusID             311 non-null    int64         
     4   GenderID                    311 non-null    int64         
     5   EmpStatusID                 311 non-null    int64         
     6   DeptID                      311 non-null    int64         
     7   PerfScoreID                 311 non-null    int32         
     8   FromDiversityJobFairID      311 non-null    int64         
     9   Salary                      311 non-null    int32         
     10  Termd                       311 non-null    int64         
     11  PositionID                  311 non-null    int32         
     12  Position                    311 non-null    object        
     13  State                       311 non-null    object        
     14  Zip                         311 non-null    int64         
     15  DOB                         311 non-null    datetime64[ns]
     16  Sex                         311 non-null    object        
     17  MaritalDesc                 311 non-null    object        
     18  CitizenDesc                 311 non-null    object        
     19  HispanicLatino              311 non-null    object        
     20  RaceDesc                    311 non-null    object        
     21  DateofHire                  311 non-null    datetime64[ns]
     22  DateofTermination           104 non-null    datetime64[ns]
     23  TermReason                  311 non-null    object        
     24  EmploymentStatus            311 non-null    object        
     25  Department                  311 non-null    object        
     26  ManagerName                 311 non-null    object        
     27  ManagerID                   303 non-null    float64       
     28  RecruitmentSource           311 non-null    object        
     29  PerformanceScore            311 non-null    object        
     30  EngagementSurvey            311 non-null    float64       
     31  EmpSatisfaction             311 non-null    int64         
     32  SpecialProjectsCount        311 non-null    int64         
     33  LastPerformanceReview_Date  311 non-null    object        
     34  DaysLateLast30              311 non-null    int64         
     35  Absences                    311 non-null    int64         
     36  Age                         311 non-null    int64         
     37  Tenure                      311 non-null    int64         
    dtypes: datetime64[ns](3), float64(2), int32(3), int64(15), object(15)
    memory usage: 88.8+ KB
    


```python
dataset.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Employee_Name</th>
      <th>EmpID</th>
      <th>MarriedID</th>
      <th>MaritalStatusID</th>
      <th>GenderID</th>
      <th>EmpStatusID</th>
      <th>DeptID</th>
      <th>PerfScoreID</th>
      <th>FromDiversityJobFairID</th>
      <th>Salary</th>
      <th>...</th>
      <th>RecruitmentSource</th>
      <th>PerformanceScore</th>
      <th>EngagementSurvey</th>
      <th>EmpSatisfaction</th>
      <th>SpecialProjectsCount</th>
      <th>LastPerformanceReview_Date</th>
      <th>DaysLateLast30</th>
      <th>Absences</th>
      <th>Age</th>
      <th>Tenure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adinolfi, Wilson  K</td>
      <td>10026</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>62506</td>
      <td>...</td>
      <td>LinkedIn</td>
      <td>Exceeds</td>
      <td>4.60</td>
      <td>5</td>
      <td>0</td>
      <td>1/17/2019</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>4528</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ait Sidi, Karthikeyan</td>
      <td>10084</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>104437</td>
      <td>...</td>
      <td>Indeed</td>
      <td>Fully Meets</td>
      <td>4.96</td>
      <td>3</td>
      <td>6</td>
      <td>2/24/2016</td>
      <td>0</td>
      <td>17</td>
      <td>48</td>
      <td>444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Akinkuolie, Sarah</td>
      <td>10196</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>64955</td>
      <td>...</td>
      <td>LinkedIn</td>
      <td>Fully Meets</td>
      <td>3.02</td>
      <td>3</td>
      <td>0</td>
      <td>5/15/2012</td>
      <td>0</td>
      <td>3</td>
      <td>35</td>
      <td>447</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alagbe,Trina</td>
      <td>10088</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>64991</td>
      <td>...</td>
      <td>Indeed</td>
      <td>Fully Meets</td>
      <td>4.84</td>
      <td>5</td>
      <td>0</td>
      <td>1/3/2019</td>
      <td>0</td>
      <td>15</td>
      <td>35</td>
      <td>5803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Anderson, Carol</td>
      <td>10069</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>50825</td>
      <td>...</td>
      <td>Google Search</td>
      <td>Fully Meets</td>
      <td>5.00</td>
      <td>4</td>
      <td>0</td>
      <td>2/1/2016</td>
      <td>0</td>
      <td>2</td>
      <td>34</td>
      <td>1884</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpID</th>
      <th>MarriedID</th>
      <th>MaritalStatusID</th>
      <th>GenderID</th>
      <th>EmpStatusID</th>
      <th>DeptID</th>
      <th>PerfScoreID</th>
      <th>FromDiversityJobFairID</th>
      <th>Salary</th>
      <th>Termd</th>
      <th>PositionID</th>
      <th>Zip</th>
      <th>ManagerID</th>
      <th>EngagementSurvey</th>
      <th>EmpSatisfaction</th>
      <th>SpecialProjectsCount</th>
      <th>DaysLateLast30</th>
      <th>Absences</th>
      <th>Age</th>
      <th>Tenure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>303.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
      <td>311.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10156.000000</td>
      <td>0.398714</td>
      <td>0.810289</td>
      <td>0.434084</td>
      <td>2.392283</td>
      <td>4.610932</td>
      <td>2.977492</td>
      <td>0.093248</td>
      <td>69020.684887</td>
      <td>0.334405</td>
      <td>16.845659</td>
      <td>6555.482315</td>
      <td>14.570957</td>
      <td>4.110000</td>
      <td>3.890675</td>
      <td>1.218650</td>
      <td>0.414791</td>
      <td>10.237942</td>
      <td>44.408360</td>
      <td>2915.569132</td>
    </tr>
    <tr>
      <th>std</th>
      <td>89.922189</td>
      <td>0.490423</td>
      <td>0.943239</td>
      <td>0.496435</td>
      <td>1.794383</td>
      <td>1.083487</td>
      <td>0.587072</td>
      <td>0.291248</td>
      <td>25156.636930</td>
      <td>0.472542</td>
      <td>6.223419</td>
      <td>16908.396884</td>
      <td>8.078306</td>
      <td>0.789938</td>
      <td>0.909241</td>
      <td>2.349421</td>
      <td>1.294519</td>
      <td>5.852596</td>
      <td>8.870236</td>
      <td>1375.284534</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10001.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>45046.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1013.000000</td>
      <td>1.000000</td>
      <td>1.120000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>26.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10078.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>55501.500000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>1901.500000</td>
      <td>10.000000</td>
      <td>3.690000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>37.000000</td>
      <td>1812.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10156.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>62810.000000</td>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>2132.000000</td>
      <td>15.000000</td>
      <td>4.280000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>43.000000</td>
      <td>3304.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10233.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>72036.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>2355.000000</td>
      <td>19.000000</td>
      <td>4.700000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>50.000000</td>
      <td>3794.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10311.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>250000.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>98052.000000</td>
      <td>39.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>20.000000</td>
      <td>72.000000</td>
      <td>6531.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preparation

Before we begin our analysis, let's prepare our data by calculating the age of employees and ensuring data types are correct.


```python
# Convert 'DOB' to datetime and calculate age
dataset['DOB'] = pd.to_datetime(dataset['DOB'])
current_year = datetime.now().year
dataset['Age'] = current_year - dataset['DOB'].dt.year

# Convert relevant columns to appropriate data types
dataset['PerfScoreID'] = dataset['PerfScoreID'].astype(int)
dataset['Salary'] = dataset['Salary'].astype(int)
dataset['PositionID'] = dataset['PositionID'].astype(int)
```

## Data Visualization

Now, let's create scatter plots to visualize the relationships between performance score and salary, age, and job level.


```python
# Performance vs. Salary
plt.figure(figsize=(10, 5))
sns.scatterplot(data=dataset, x='PerfScoreID', y='Salary')
plt.title('Performance Score vs Salary')
plt.show()
```


    
![png](/images/HR/output_10_0.png)
    


- The plot shows a series of discrete vertical clusters, corresponding to different performance scores.
- A positive correlation appears to be present, indicating that higher performance scores may be associated with higher salaries.
- The highest salaries are concentrated at the highest performance score (4.0), suggesting a possible reward system for top performers.
- There is a wide range of salaries within each performance score category, especially at the higher performance scores.


```python
# Performance vs. Age
plt.figure(figsize=(10, 5))
sns.scatterplot(data=dataset, x='PerfScoreID', y='Age')
plt.title('Performance Score vs Age')
plt.show()
```


    
![png](/images/HR/output_12_0.png){:.centered}
    


- The distribution of ages is relatively uniform across different performance scores.
- There doesn't seem to be a clear trend or correlation between age and performance score, suggesting that performance may not be directly influenced by an employee's age.
- Employees of a wide range of ages can be found at each performance score level, showing diversity in age among different performance categories.


```python
# Performance vs. Job Level
plt.figure(figsize=(10, 5))
sns.scatterplot(data=dataset, x='PerfScoreID', y='PositionID')
plt.title('Performance Score vs Job Level')
plt.show()
```


    
![png](/images/HR/output_14_0.png){:.centered}
    


- Higher job levels appear more frequently at higher performance scores, suggesting a potential link between job level and performance.
- The most populated vertical cluster is at the highest performance score (4.0), which may indicate that higher job levels have a higher concentration of top performance ratings.
- There is a noticeable absence of lower job levels at the highest performance score, which could imply that such levels have a performance cap or that promotion to a higher job level is performance-dependent.

Overall, these plots suggest that while salary and job level have some association with performance scores, age does not show a significant relationship with performance. The findings indicate that the company might reward high performance with higher salaries and job levels, but performance evaluation is age-agnostic. These visual insights should be further investigated with statistical analyses to understand the strength and significance of these relationships.

## Correlation Analysis

To quantify the relationships, we will calculate the correlation coefficients.


```python
# Calculating correlation coefficients
salary_corr = dataset['PerfScoreID'].corr(dataset['Salary'])
age_corr = dataset['PerfScoreID'].corr(dataset['Age'])
job_level_corr = dataset['PerfScoreID'].corr(dataset['PositionID'])

# Print the correlation coefficients
print(f"Correlation between Performance Score and Salary: {salary_corr}")
print(f"Correlation between Performance Score and Age: {age_corr}")
print(f"Correlation between Performance Score and Job Level: {job_level_corr}")
```

    Correlation between Performance Score and Salary: 0.1309025823175193
    Correlation between Performance Score and Age: 0.07920301894181259
    Correlation between Performance Score and Job Level: 0.005226508043668236
    

A correlation coefficient of **0.1309** indicates a positive but weak relationship between Performance Score and Salary. This suggests that while there may be a tendency for higher performance scores to correspond to higher salaries, the relationship is not strong. Many other factors likely contribute to determining salary that are not captured by performance score alone.


The correlation coefficient of **0.0792** is closer to zero, suggesting a very weak positive relationship between Performance Score and Age. This aligns with the visual observation from the scatter plot, where age did not appear to have a significant impact on performance score. Performance seems to be relatively independent of age.


With a correlation coefficient of **0.0052**, there is virtually no linear relationship between Performance Score and Job Level. This coefficient is very close to zero, which suggests that job level does not predict performance score and vice versa, at least not linearly. This might be surprising given the visual cluster of higher job levels at the top performance score, but it indicates that across all job levels, performance scores are varied and not systematically higher with increasing job level.

These correlation coefficients highlight the importance of considering multiple factors when analyzing employee performance and compensation. Although there may be a perceived relationship, the actual correlation may be weak, emphasizing that performance scores are influenced by a complex interplay of various factors beyond just salary, age, or job level.

## Employee Satisfaction Levels Across Different Departments

Let's examine if there's a difference in employee satisfaction levels across various departments.

### Data Visualization


```python
# Grouping by Department and calculating average satisfaction
department_satisfaction = dataset.groupby('Department')['EmpSatisfaction'].mean()

# Creating a bar chart with figure size specified for better visibility
plt.figure(figsize=(12, 6))
bars = plt.bar(department_satisfaction.index, department_satisfaction.values)

# Adding the value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

plt.title('Average Employee Satisfaction by Department')
plt.ylabel('Average Satisfaction')
plt.xticks(rotation=45, ha='right')  # Rotate labels to avoid overlap
plt.show()
```


    
![png](/images/HR/output_21_0.png){:.centered}
    


- All departments have a relatively high level of average employee satisfaction, with all scores above **3.0** on a scale that appears to go up to **4.0**.
- The Admin Offices have the lowest average satisfaction score among the five departments, but the score is still above **3.0**, indicating generally positive satisfaction.
- The Executive Office has a slightly higher average satisfaction score than the Admin Offices but is still lower than the other three departments.
- IT/IS, Production, and Software Engineering departments have very similar levels of average employee satisfaction, all close to **4.0**, which suggests that employees in these departments are the most satisfied overall.
- The Software Engineering department has the highest average satisfaction score, albeit by a narrow margin.

Based on this chart, one can conclude that while employee satisfaction is generally high across the company, there are slight variations between departments. The reasons behind the lower satisfaction in the Admin Offices compared to the other departments could be a point of interest for further investigation. Additionally, the factors contributing to the high satisfaction in Software Engineering might be explored for potential adoption in other departments.

## Job Level and Likelihood to Leave the Company

We will investigate how an employee's job level relates to their likelihood of leaving the company.

### Data Analysis and Visualization



```python
# Calculating attrition rate by job level
attrition_rate = dataset[dataset['Termd'] == 1].groupby('PositionID').size() / dataset.groupby('PositionID').size()

# Creating a bar chart
plt.figure(figsize=(10, 5))
attrition_rate.plot(kind='bar')
plt.title('Attrition Rate by Job Level')
plt.ylabel('Attrition Rate')
plt.show()
```


    
![png](/images/HR/output_24_0.png){:.centered}
    


- The attrition rate varies significantly across different job levels.
- Lower job levels (as indicated by the lower PositionID numbers) generally have lower attrition rates, with some fluctuation.
- As the PositionID increases, indicating higher job levels, there is a general trend of increasing attrition rate, with some positions having notably higher rates than others.
- The highest attrition rate is seen at the highest PositionID shown on the chart, which is nearly **1.0**. This could suggest that employees at this job level are either all or almost all leaving the company.

Based on these observations, it can be inferred that job level may be a factor in an employee's decision to stay with or leave the company. The reasons behind the higher attrition rate at the highest job level could be varied, ranging from career advancement opportunities elsewhere to possible dissatisfaction or restructuring at higher organizational levels. It may be beneficial for the company to investigate the specific causes of attrition at higher job levels to develop strategies to retain key talent.

## Impact of Tenure on Salary and Performance Rating

Next, we'll explore how the length of time an employee has been with the company impacts their salary and performance rating.

### Data Preparation


```python
# Calculate tenure
dataset['DateofHire'] = pd.to_datetime(dataset['DateofHire'])
dataset['DateofTermination'] = pd.to_datetime(dataset['DateofTermination'])
dataset['Tenure'] = dataset.apply(lambda x: (x['DateofTermination'] - x['DateofHire']).days if pd.notna(x['DateofTermination']) else (datetime.now() - x['DateofHire']).days, axis=1)
```

### Data Visualization


```python
# Tenure vs. Salary
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataset, x='Tenure', y='Salary')
plt.title('Tenure vs Salary')
plt.show()
```


    
![png](/images/HR/output_29_0.png){:.centered}
    


- The plot shows a wide range of salaries across different tenure lengths.
- There is no distinct upward trend that suggests a strong positive correlation between tenure and salary. While some employees with longer tenure have higher salaries, there is considerable variation, and some long-tenured employees have salaries similar to those with shorter tenures.
- There are several outliers, particularly employees with higher salaries, which do not follow a clear trend related to tenure.
- The majority of data points are clustered at the lower end of the salary spectrum, regardless of tenure.


```python
# Tenure vs. Performance Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataset, x='Tenure', y='PerfScoreID')
plt.title('Tenure vs Performance Score')
plt.show()
```


    
![png](/images/HR/output_31_0.png){:.centered}
    


- Performance scores are distributed across various tenure lengths with a large concentration of scores around the **3.0** mark.
- There is no clear trend indicating that longer tenure results in higher performance scores. Performance scores seem to be consistent across different tenure lengths.
- There is a ceiling effect visible with performance scores, where scores do not exceed **4.0**. This could be the maximum score achievable in the performance rating system.
- The distribution suggests that performance evaluations are not directly tied to the length of tenure, as employees with varying lengths of tenure have similar performance scores.

These findings suggest that while there may be some relationship between tenure and salary, it is not strongly linear, and other factors likely play a significant role in determining salary. Similarly, tenure does not appear to be a strong predictor of performance score. This could imply that the company's performance evaluations are based on criteria other than tenure, or that employees reach a performance plateau regardless of how long they have been with the company.

## Reasons for Employee Attrition and Patterns

Finally, we'll look at the most common reasons for employee attrition and see if there are any patterns based on age, tenure, or performance rating.

### Data Visualization for Reasons of Attrition


```python
# Counting the reasons for attrition
attrition_reasons = dataset[dataset['Termd'] == 1]['TermReason'].value_counts()

# Creating a bar chart
plt.figure(figsize=(10, 5))
attrition_reasons.plot(kind='bar')
plt.title('Reasons for Employee Attrition')
plt.xlabel('Reasons')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')  # Rotate labels to avoid overlap
plt.show()
```


    
![png](/images/HR/output_34_0.png){:.centered}
    


- The most common reason for employee attrition is **another position**, indicating that a significant number of employees leave for jobs at other companies.
- The second most frequent reason is **unhappy**, suggesting that job dissatisfaction is a major factor in employees' decision to leave.
- **More money** and **career change** follow as the next most common reasons, which could imply that financial incentives and professional growth opportunities elsewhere are influencing factors in attrition.
- Other notable reasons include **hours**, **attendance**, and **return to school**, but these are less frequent compared to the top reasons.
- Reasons such as **medical issues**, **relocating out of area**, **military**, **retiring**, **maternity leave - did not return**, **performance**, and **no-call, no-show** are the least common, indicating that they are not the primary drivers of employee turnover in this dataset.
- The variety of reasons suggests that attrition is a multifaceted issue, with a range of personal and professional factors contributing to employees' decisions to leave.

These insights can help the company address the underlying causes of attrition. For example, improving job satisfaction and offering competitive compensation may reduce turnover. Additionally, providing clear career progression paths could potentially retain employees looking for a career change. Understanding these reasons in more depth could guide targeted retention strategies.

### Analyzing Patterns of Attrition


```python
# Attrition by Age
plt.figure(figsize=(10, 6))
sns.boxplot(x='Termd', y='Age', data=dataset)
plt.title('Attrition by Age')
plt.show()
```


    
![png](/images/HR/output_37_0.png){:.centered}
    


- The age range of employees who have not left the company (Termd = 0) is slightly lower with a median age around the late 40s.
- The age distribution for those who have left the company (Termd = 1) is broader, with a higher median age around the early 50s.
- There are outliers in both categories, with some older employees who have not left the company and some who have.


```python
# Attrition by Tenure
plt.figure(figsize=(10, 6))
sns.boxplot(x='Termd', y='Tenure', data=dataset)
plt.title('Attrition by Tenure')
plt.show()
```


    
![png](/images/HR/output_39_0.png){:.centered}
    


- Employees who have not left the company tend to have a wide range of tenure, with a median tenure that appears to be around 3000 days, suggesting long-term employment.
- The tenure of employees who have left is generally lower, with a median around 2000 days, indicating they leave the company earlier in their tenure.
- The range of tenure is narrower for those who have left, which suggests there's less variability in tenure among employees who decide to leave.


```python
# Attrition by Performance Score
plt.figure(figsize=(10, 6))
sns.boxplot(x='Termd', y='PerfScoreID', data=dataset)
plt.title('Attrition by Performance Score')
plt.show()
```


    
![png](/images/HR/output_41_0.png){:.centered}
    


- The performance scores for employees who have not left the company are distributed mostly at the higher end of the scale, with the median score close to **3.0**.
- For employees who have left, the performance scores are also high, with a median score similar to those who haven't left, which suggests that performance scores alone are not a strong predictor of whether an employee will leave.
- There are a few outliers on both sides, indicating some employees with lower performance scores have not left, and some with higher scores have left.

Overall, these box plots suggest that age and tenure may have some influence on attrition, with employees who are older and have shorter tenure being more likely to leave. Performance score does not appear to be a decisive factor in attrition, as scores are relatively high for both those who have and have not left the company. It's important to note that box plots show the distribution of a dataset and can reveal the central tendency and dispersion, but they do not demonstrate causation. Further analysis would be necessary to understand the reasons behind these trends.

In conclusion, while performance seems to be recognized in terms of salary to some degree, there are clear indications that the company might benefit from examining its compensation strategies, job satisfaction drivers, and career development opportunities to address attrition causes. The lack of strong correlation between performance and job level suggests that career progression might not be entirely merit-based or that the evaluation system spreads scores evenly across roles. Addressing these findings could lead to improved employee retention, better job satisfaction, and a more transparent and effective performance evaluation system.
