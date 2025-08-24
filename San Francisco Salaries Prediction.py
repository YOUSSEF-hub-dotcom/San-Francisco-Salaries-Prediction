from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import value_counts
from scipy.stats import alpha
pd.set_option('display.width',None)
df = pd.read_csv("C:\\Users\\youss\\Downloads\\Salaries.csv")
print(df.head(50))
print("------------------------------------")
print("==========>>> Basic Function:")
print("number of rows and column:")
print(df.shape)
print("Data Type in data:")
print(df.dtypes)
print("The name of columns:")
print(df.columns)
print("The information about data:")
print(df.info())
print("Statistical operations:")
print(df.describe().round())
print("number of missing values:")
print(df.isnull().sum())
print("number of frequency:")
print(df.duplicated().sum())

print("------------------------")
print("============>>> Cleaning Data:")
missing_percentage = df.isnull().mean() * 100
print("The percentage of missing values in every column:\n",missing_percentage)
print("Missing Values before cleaning :")
print(df.isnull().sum())
print("The miss value in BasePay is 3.4% so we use fillna")
df['BasePay'] = df['BasePay'].fillna(df['BasePay'].median())
print("The miss value in OvertimePay is 9.2% so we use fillna")
df['OvertimePay'] = df['OvertimePay'].fillna(df['OvertimePay'].median())
print("The miss value in Benefits is 24.3% so we use fillna")
df['Benefits'] = pd.to_numeric(df['Benefits'],errors='coerce')
df['Benefits'] = df['Benefits'].fillna(df['Benefits'].median())
print("The Benefits,OvertimePay,and BasePay column contain miss value ?")
print(df[['BasePay','OvertimePay','Benefits']].isnull().sum())
print("The miss value in Status is 74% so we use drop")
df = df.drop(columns=['Status'])
print("The miss value in Notes is 100% so we use drop")
df = df.drop(columns=['Notes'])
print("The Status,Notes columns contain more miss value so we remove them")
print("Missing Values after cleaning :")
print(df.isnull().sum())
sns.heatmap(df.isnull())
plt.title('The missing value in San Francisco')
plt.show()

print("------------------------------------")
print("-----------------------")
print("=========>>> Exploration Data Analysis :")
print("Select The first row of the data")
print(df.iloc[0,:])
print("Select The first column of data")
print(df.iloc[:,0])
print("Select The first 3 rows")
print(df.iloc[:3,:])
print("Select The last column")
print(df.iloc[:,-1])
print("Fancy Indexing")
print(df.iloc[[1,5,50],:])      # rows and column
print("Select the last 10 rows")
print(df.iloc[-10:,:])
print("The average BasePay:")
df['BasePay'] = pd.to_numeric(df['BasePay'],errors='coerce')
print(df['BasePay'].mean())
print("The highest amount of OvertimePay in the dataset:")
df['OvertimePay'] = pd.to_numeric(df['OvertimePay'],errors='coerce')
print(df['OvertimePay'].max())
print("what is the job title of JOSEPH DRISCOLL?")
print(df[df['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle'])
print("How much does JOSEPH DRISCOLL make (including benefits)?")
print(df[df['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits'])
print("The name of the highest paid person (including benefits)?")
print(df[df['TotalPayBenefits']== df['TotalPayBenefits'].max()]['EmployeeName'])
print("The name of the lowest paid person (including benefits)?")
print(df[df['TotalPayBenefits']== df['TotalPayBenefits'].min()]['EmployeeName'])
print("The average BasePay of all employees per year?(2011 -2014)") # time series
print(df.groupby('Year')['BasePay'].mean())
print('How many unique job titles?')
print(df['JobTitle'].nunique())
print('what are the top 5 most common jobs?')
print(df['JobTitle'].value_counts().head(5))
print('what are the least 10 most common jobs?')
print(df['JobTitle'].value_counts().tail(10))
print("How many Job Titles were represented by only one person in 2013")
print((df[df['Year'] == 2013] ['JobTitle'].value_counts() == 1).sum())
print("How many Job Titles were represented by only one person per Year")
print((df.groupby('Year')['JobTitle'].value_counts()==1).sum())
print("How many people have the word Chief title?")
def chief(string):
    if 'chief' in (string.lower()):
        return True
    else:
        return False
print((df['JobTitle'].apply(lambda x:chief(x))).sum())
print("Is There a correlation between length of the JobTitle string and Salary")
df['n_titles'] = df['JobTitle'].apply(len)
print(df['n_titles'])
print(df['n_titles'].corr(df['TotalPayBenefits']))

print("------------------------------------")
print("==============>>> Visualization")
# Index(['Id', 'EmployeeName', 'JobTitle', 'BasePay', 'OvertimePay', 'OtherPay',
#        'Benefits', 'TotalPay', 'TotalPayBenefits', 'Year', 'Notes', 'Agency',
#        'Status']
#BarPlot
plt.figure(figsize=(10,6))
sns.barplot(x='Agency',y='TotalPay',data=df,hue='Agency',palette='viridis',legend=False)
#plt.xticks(rotation=45,ha='right')
plt.title("Total Pay By Agency",fontsize=16,pad=15)
plt.xlabel("Agency",fontsize=12)
plt.ylabel("Total Pay",fontsize=12)
plt.tight_layout()
plt.show()
#----------------------------------------------------
#PieChart
plt.figure(figsize=(8,8))
job_counts = df['JobTitle'].value_counts().head(5)
plt.pie(job_counts,labels=job_counts.index,autopct='%1.1f%%',colors=sns.color_palette('pastel'),startangle=90)
plt.title("Distribution of Benefits By Top 5 Job Titles",fontsize=16,pad=20)
plt.axis('equal')
plt.show()
#----------------------------------------------------
#ScatterPlot
plt.figure(figsize=(10,6))
sns.scatterplot(x='BasePay',y='OvertimePay',data=df,hue ='Agency' ,size='Benefits',palette='deep',alpha=0.6)
plt.title("Base Pay Vs OverTime Pay By Agency",fontsize=16,pad=15)
plt.xlabel("Base Pay",fontsize=12)
plt.ylabel("OverTime Pay",fontsize=12)
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.tight_layout()
plt.show()
#----------------------------------------------------
print("------------- Machine Learning Model ----------------------")
df.drop(['EmployeeName','JobTitle','Year','Agency'], axis=1, inplace=True)

df = df.replace("Not Provided", np.nan)
df = df.fillna(0)

print("=========>>> Building Model :")

X = df[['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']]
y = df['TotalPay']

print("X shape:", X.shape)
print("y shape:", y.shape)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("=========>>> Model Training and Prediction >> LinearRegression:")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=========>>> Model Evaluation >> LinearRegression:")

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)  # 99%
print("model.score:", model.score(X, y))

print("Sample Predictions:")
print("Actual:", list(y_test[:5]))
print("Predicted:", list(y_pred[:5]))

print("=========>>> Model Training and Prediction >> RandomForestRegressor:")

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_predict = model_rf.predict(X_test)

print("=========>>> Model Evaluation >> RandomForestRegressor:")
print("Mean Squared Error:", mean_squared_error(y_test, y_predict))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_predict))
print("R2 Score:", r2_score(y_test, y_predict))
print("model.score (Train) :", model_rf.score(X, y))
print("model.score (Test):", model_rf.score(X_test, y_test))

