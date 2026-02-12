import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
df=pd.read_csv("Telco-Customer-Churn.csv").copy()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges']=df['TotalCharges'].fillna(0)
target=df['Churn'].map({'Yes':1,'No':0})
features=df.drop(['Churn','customerID','gender'],axis=1)

categorical=['Partner','PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 
              'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaperlessBilling', 'PaymentMethod']
numerical=['SeniorCitizen','tenure','MonthlyCharges',
           'TotalCharges']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessing = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical),
        ('cat', cat_pipeline, categorical)
    ]
)
pipeline1 = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('classifier',RandomForestClassifier(n_estimators=1000,random_state=42,class_weight='balanced')
)
])

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.3,random_state=42)

pipeline1.fit(x_train,y_train)
y_pred = pipeline1.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(accuracy*100)

result=pd.DataFrame({
    "Actual":y_test,
    "Prediction":y_pred
})
result.to_csv("Output.csv",index=False)
print("Output.csv save...")




