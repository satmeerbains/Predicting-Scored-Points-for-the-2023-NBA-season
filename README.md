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

