# Big-Data---classifier
The aim of the project is to find the best classifier for classifying the positions of football players based on their match statistics from the 2021-2022 season. To achieve this, the "2021-2022 Football Player Stats" dataset was utilized, containing over 2500 records and 143 columns (features). The analysis focused on the average player statistics per match from the top five leagues (Premier League, Ligue 1, Bundesliga, Serie A, La Liga). You can see the process and visualization here: [REPORT](fppc_report.pdf). 
I used:
- Pandas: Used for reading data from CSV files and data manipulation. Initial data analysis and processing.
- Seaborn and Matplotlib: Utilized for data visualization and generating plots.
- Scikit-learn: Employed for data standardization, dimensionality reduction using PCA, and classification using RandomForestClassifier, KNeighborsClassifier, SVC.
- StandardScaler: Used to standardize data before applying classification algorithms.
- PCA (Principal Component Analysis): Utilized for reducing data dimensions.
- RandomForestClassifier.feature_importances_: Utilized to obtain sorted features based on their influence on the ability to classify the target variable.
- Cross_val_score: Used to evaluate classifier quality based on F1-Score.
- Classification_report: Used for detailed evaluation of the best classifier.
- Confusion_matrix: Employed to assess the accuracy of conducted classification.
- Train_test_split: Used to split the dataset into training and testing sets.
