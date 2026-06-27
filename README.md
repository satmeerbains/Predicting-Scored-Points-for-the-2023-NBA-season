# 🏀 Predicting Scored Points for the 2023 NBA season🏀 
## Overview
Predicting the points scored for each player in the NBA is very important in basketball analytics. It is a crucial performance metric that allows coaches and analysts to assess a player's scoring ability and overall offensive impact to the team. Understanding a players scoring potential aids in strategic decision-making during games, player selection, and talent scouting. In this notebook, we will employ various machine learning techniques to predict players' point contributions using data from the 2023 season.

<h3> Table of Contents </h3>
<ul style="list-style-type: none; padding-left: 0;">
    1. Import libraries</li> <br>
    2. Data Exploration</li> <br>
    3. Data Visualization</li> <br>
    4. Features Selection</li> <br>
    5. Modeling:
        <ul style="list-style-type: none; padding-left: 20px;">
            <span style="margin-left: -10px;">&#8226;</span> Linear Regression</li> <br>
            <span style="margin-left: -10px;">&#8226;</span> KNeighbors Regressor</li> <br>
            <span style="margin-left: -10px;">&#8226;</span> Decision Tree Regressor</li> <br>
            <span style="margin-left: -10px;">&#8226;</span> Random Forest Regressor</li> <br>
        </ul>
    </li>
    6. Results</li>
</ul>
</span>
    <span>The machine learning techniques used to predict basketball points are: Linear Regression, K-Nearest Neighbors (KNN) Regressor, Decision Tree Regressor (DT), and Random Forest Regressor (RFR). Each model is utilized to predict players' total points based on various performance metrics, such as minutes played, field goals made, free throws made, assists, steals, blocks, and other relevant statistics.</span>
    <br>
    <br>
    <span>By employing these regression models, we aim to understand how they perform in predicting basketball points and compare their respective predictive capabilities. Through this analysis, we can gain valuable insights into the strengths and weaknesses of each model and identify the most suitable model for predicting basketball points in this specific dataset.</span>
    <br>
    <br>

**Tech stack:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly

    
<h3> Dataset </h3>
The data used was player data from the National Basketball Association from the 2022-2023 season. 

## 1. Importing Libraries
Libraries used in this notebook inlcude pandas, numpy, plotly, sklearn, and seaborn

## 2. Data Exploration
The dataset is loaded from `2023_nba_player_stats.csv` and contains per-player season stats for the 2023 NBA season.

**Steps:**
- Checked dataset shape and duplicate rows
- Renamed columns to clearer, more descriptive names (e.g. `PTS` → `Total_Points`, `GP` → `Games_Played`)
- Reviewed data types and summary statistics (`df.info()`, `df.describe()`)
- Checked for missing values
- Found 5 missing values in the `Position` column — all identified as shooting guards based on research, and filled with `'SG'`
- 
The following is a snapshot of the original dataframe
<img width="2674" height="656" alt="image" src="https://github.com/user-attachments/assets/6fbfc74d-edac-4846-8467-94f4b6eda9ae" />

Updates made:
 <ul style="list-style-type: none; padding-left: 20px;">
            <span style="margin-left: -10px;">&#8226;</span> Columns were renamed due to the ambiguity of the naming convention in the original dataframe</li> <br>
            <span style="margin-left: -10px;">&#8226;</span> The 'Position' feature had missing values. Based on research all 5 players that have this missing value are shooting guards the null values were updated to be 'SG'
        </ul>
    </li>

The following shows the data description 
<img width="2548" height="526" alt="image" src="https://github.com/user-attachments/assets/6f3330ee-368b-4082-aa44-d901b94d9d52" />

## 3. Data Visualization
The following are visualizations generated based on the dataset.
<img width="1342" height="450" alt="image" src="https://github.com/user-attachments/assets/db846a85-aee7-4592-ab80-0d2fac5ab1c3" />
<img width="1342" height="450" alt="newplot (2)" src="https://github.com/user-attachments/assets/56e4224b-2eed-4aa2-b066-51d641bddb12" />
<img width="1342" height="450" alt="newplot (1)" src="https://github.com/user-attachments/assets/31f9fc0d-4bed-455d-b720-6df5e4685b27" />
<img width="1342" height="450" alt="newplot (12)" src="https://github.com/user-attachments/assets/27a3f666-f2a2-41f7-8695-66a3191cac6e" />
<img width="950" height="500" alt="newplot (11)" src="https://github.com/user-attachments/assets/8af0fe8f-34cc-49c0-8a97-a4e0596d8146" />
<img width="1342" height="450" alt="newplot (10)" src="https://github.com/user-attachments/assets/0f19152f-7bf2-4894-b5cf-2b46dfe8d8ce" />
<img width="1342" height="450" alt="newplot (9)" src="https://github.com/user-attachments/assets/76b83ac1-8685-4a3b-a4bb-3b7e76c02d07" />
<img width="1342" height="450" alt="newplot (8)" src="https://github.com/user-attachments/assets/2f3c6e94-7f0a-4f8c-9c7a-df79ea6e4752" />
<img width="1342" height="450" alt="newplot (7)" src="https://github.com/user-attachments/assets/36c2097b-f75d-4e15-bc3f-13edcf0342c3" />
<img width="1342" height="450" alt="newplot (6)" src="https://github.com/user-attachments/assets/5abc0fd0-f951-42ea-9a9a-595f5704e935" />
<img width="1342" height="450" alt="newplot (5)" src="https://github.com/user-attachments/assets/a98426f9-ccb3-42ad-8312-7e4e17cf38c9" />
<img width="1342" height="450" alt="newplot (4)" src="https://github.com/user-attachments/assets/955560c5-a148-4671-a8c9-f73c6cfcedd1" />
<img width="1342" height="450" alt="newplot (3)" src="https://github.com/user-attachments/assets/0ef213fb-23fd-4189-ad17-a2eea8c5bd00" />
<img width="1342" height="450" alt="newplot" src="https://github.com/user-attachments/assets/3b493834-f06d-42e0-ab5a-62dba791ac98" />
<img width="1342" height="450" alt="newplot (2)" src="https://github.com/user-attachments/assets/cb2d110f-d3af-4e3b-b443-e8e4d788480c" />
<img width="1342" height="450" alt="newplot (1)" src="https://github.com/user-attachments/assets/54c992a1-6e5c-4dbb-ae14-fa3845b8aa48" />
<img width="1342" height="450" alt="newplot (1)" src="https://github.com/user-attachments/assets/f268a879-8eb7-46ca-a128-439ced8b6f6a" />
<img width="1342" height="450" alt="newplot" src="https://github.com/user-attachments/assets/ea800dbd-83b8-4123-915b-526b148a174c" />
<img width="1000" height="800" alt="newplot (2)" src="https://github.com/user-attachments/assets/4584feec-e06f-4789-98c4-2dfb04f5a44f" />


## 4. Feature Selection
To reduce noise and prevent multicollinearity from distorting the regression models, two preprocessing steps were applied prior to training.

First, rows with implausible or invalid shooting statistics were removed: field goal percentage greater than 90% or equal to 0%, three-point percentage greater than 90% or equal to 0%, and free throw percentage equal to 0%. These thresholds were chosen because such values almost always result from extremely low attempt counts rather than genuine shooting performance, and including them would distort the relationship between shooting efficiency and scoring output.

Second, a set of features was dropped based on the correlation heatmap generated during exploratory analysis. Identifying columns, including `Player_Name`, `Position`, and `Team_Abbreviation`, were excluded since they carry no predictive signal in their raw form. Additionally, several stat pairs that are mathematically or practically redundant — such as `Field_Goals_Made`/`Field_Goals_Attempted`, the three-point shooting columns, and offensive/defensive rebound splits — were removed to mitigate multicollinearity, which can inflate coefficient variance in linear models and obscure feature importance in tree-based models.

The resulting dataset was split into training and test sets using an 80/20 ratio with a fixed random state of 42 to ensure reproducibility, with `Total_Points` defined as the target variable.

## 5. Modeling

Four regression algorithms were evaluated to predict total points scored: Linear Regression, K-Nearest Neighbors, Decision Tree, and Random Forest. For each model, a grid search over train/test split ratios and random states was first conducted to identify a favorable data partition, followed by targeted hyperparameter tuning via `GridSearchCV` with 5-fold cross-validation.

**Linear Regression** served as the baseline model. A sweep across test sizes (15–30%) and six random states identified a 20% test split with `random_state=43` as optimal, yielding an R² of 0.973 on the held-out test set. The strength of this result suggests that the relationship between the selected features (particularly minutes played, shooting volume, and efficiency metrics) and total points is largely linear in nature.

**K-Nearest Neighbors** was tuned over the number of neighbors, distance weighting scheme, and distance metric (Manhattan vs. Euclidean). The best configuration (`n_neighbors=7`, `weights='distance'`, `p=1`) achieved a cross-validated R² of 0.907, with a test set R² of 0.921. The comparatively lower performance suggests that KNN's reliance on local neighborhood structure is less effective here than the global patterns captured by linear and ensemble methods.

**Decision Tree Regressor** was tuned across maximum depth, minimum samples per split, and minimum samples per leaf to control overfitting. The best parameters achieved a cross-validated R² of approximately 0.947, with the final model — evaluated using an independent test split tuning pass — reaching an R² of 0.971 on a 10% holdout set.

**Random Forest Regressor** was tuned over the number of estimators (80–160) and maximum tree depth (1–19). The best configuration achieved a cross-validated R² of 0.947 on a 10% test split, with the final model evaluated at an R² of 0.977 — the strongest result among all four approaches. As an ensemble of decision trees, Random Forest benefits from variance reduction through bagging, which appears to translate into a meaningful improvement over the single-tree approach.
markdown## Results

| Model | R² Score |
|---|---|
| Linear Regression | 0.973 |
| K-Nearest Neighbors | 0.921 |
| Decision Tree | 0.955 |
| **Random Forest** | **0.977** |

Across all four models, total points scored proved to be highly predictable from the selected feature set, with every approach exceeding an R² of 0.90. Linear Regression's strong performance (0.973) indicates that the underlying relationship between playing time, shot volume, and scoring is close to linear once outliers and redundant features are removed. Random Forest improved marginally on this baseline (0.977), benefiting from its capacity to model non-linear interactions and feature dependencies that a purely linear model cannot capture, while Decision Tree (0.955) and KNN (0.921) trailed behind — the former due to higher variance from relying on a single tree structure, and the latter due to its sensitivity to local feature-space density rather than global trends.

Diagnostic plots comparing actual versus predicted points for the final Random Forest model — including a scatter plot, an overlaid distribution histogram, a residual plot, and a predicted-vs-true line plot — confirmed that predictions tracked closely with actual values across the full scoring range, with residuals distributed roughly symmetrically around zero and no strong evidence of systematic bias at either the low or high end of the scoring spectrum.

**Conclusion:** Random Forest Regression was selected as the final model, offering the best balance of predictive accuracy and robustness to overfitting, with an R² of 0.977 on the test set.
