# 🏀 Predicting Scored Points for the 2023 NBA season🏀 
Predicting the points scored for each player in the NBA is very important in basketball analytics. It is a crucial performance metric that allows coaches and analysts to assess a player's scoring ability and overall offensive impact to the team. Understanding a players scoring potential aids in strategic decision-making during games, player selection, and talent scouting. In this notebook, we will employ various machine learning techniques to predict players' point contributions using data from the 2023 season.</span>
    <br>
    <br>
    <span>The machine learning techniques used to predict basketball points are: Linear Regression, K-Nearest Neighbors (KNN) Regressor, Decision Tree Regressor (DT), and Random Forest Regressor (RFR). Each model is utilized to predict players' total points based on various performance metrics, such as minutes played, field goals made, free throws made, assists, steals, blocks, and other relevant statistics.</span>
    <br>
    <br>
    <span>By employing these regression models, we aim to understand how they perform in predicting basketball points and compare their respective predictive capabilities. Through this analysis, we can gain valuable insights into the strengths and weaknesses of each model and identify the most suitable model for predicting basketball points in this specific dataset.</span>
    <br>
    <br>

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

    
<h3> Dataset </h3>
The data used was player data from the National Basketball Association from the 2022-2023 season. 

## 1. Importing Libraries
Libraries used in this notebook inlcude pandas, numpy, plotly, sklearn, and seaborn

## 2. Data Exploration
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

## 2. Data Visualization
