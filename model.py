import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
import joblib

warnings.filterwarnings('ignore')


df=pd.read_csv('Customer_Churn.csv')
data=df.copy()

data['TotalCharges']=df['TotalCharges'].replace(" ","0.0")
data['TotalCharges']=data['TotalCharges'].astype('float')
print(data['TotalCharges'].dtype)
data[data['TotalCharges']==0.0]

#dropping the customerID column from the data
data.drop(columns=['customerID'],inplace=True)

#train_test_split
X=data.drop(columns=['Churn'],axis='columns')
Y=data['Churn']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,stratify=Y,test_size=0.2,shuffle=True)

le=LabelEncoder()
y_train=le.fit_transform(Y_train)
y_test=le.transform(Y_test)


numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols

ordinal_columns=['Contract']
categorical_columns=X_train.select_dtypes( include=['object']).columns.tolist()#defining manually as there is just one
print(len(categorical_columns))

categorical_columns=X_train.select_dtypes( include=['object']).columns.tolist()
ord_remove=ordinal_columns[0]
if ord_remove in categorical_columns:
    categorical_columns.remove(ord_remove)
    
contract_order=[['Month-to-month', 'One year', 'Two year']]

feat_transform=ColumnTransformer(transformers=[('minmaxscaling',MinMaxScaler(),numerical_cols)
                                               ,('ordinalenc',OrdinalEncoder(categories=contract_order),ordinal_columns)
                                               ,('onehotenc',OneHotEncoder(handle_unknown='ignore'),categorical_columns)])

X_train_transformed=feat_transform.fit_transform(X_train)
X_test_transformed = feat_transform.transform(X_test)


from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=42)
x_train_resampled,y_train_resampled=sm.fit_resample(X_train_transformed,y_train)

logreg = LogisticRegression(random_state=42, max_iter=5000)
param_dist = {
    'C': uniform(0.01, 10) ,
    'class_weight': ['balanced']     
}
random_search = RandomizedSearchCV(logreg, param_distributions=param_dist,n_iter=20, scoring='f1', cv=5,n_jobs=-1,random_state=42)
random_search.fit(x_train_resampled, y_train_resampled)
best_logreg = random_search.best_estimator_
print(random_search.best_params_)

y_pred_logreg = best_logreg.predict(X_test_transformed)

joblib.dump(best_logreg,"output_models/logreg_model.sav")
joblib.dump(feat_transform,"output_models/transformer.sav")
joblib.dump(le,"output_models/label_encoder.sav")