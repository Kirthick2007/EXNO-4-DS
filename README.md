# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

```python

```

# CODING AND OUTPUT:
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/7e38e444-fafd-4307-8b05-22b8b2f9322f)
```python
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/b66d74fc-d467-4632-a462-974c4d636c38)

```python
df.dropna()
```
![image](https://github.com/user-attachments/assets/cbd97b12-58fd-4f86-887e-8e63bd550a23)

```python
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/40c383f6-d2ee-448b-a307-46c723ac18e0)

```python
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![image](https://github.com/user-attachments/assets/90dd104c-06e7-4dd3-97a4-076b52d288cb)


```python
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

![image](https://github.com/user-attachments/assets/4ae8b6aa-5406-497e-87b3-0ae13d18873b)

```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/71d8ed19-45db-4221-959e-8b8667994bf6)

```python
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```

![image](https://github.com/user-attachments/assets/77d8ed34-b872-44f4-bcc5-0760756fa2de)

```python
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```

![image](https://github.com/user-attachments/assets/bd73f652-9071-4377-95e4-576829b57edf)

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
```

![image](https://github.com/user-attachments/assets/55fc3392-c003-4c2e-8e05-2f8e784f83f0)

```python
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```

![image](https://github.com/user-attachments/assets/c983340d-c4bf-4ac1-b166-d5539371e82d)

```python
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```

![image](https://github.com/user-attachments/assets/04e1d35a-a413-4c42-8d8d-2469fd9f5ee7)

```python
df
```

![image](https://github.com/user-attachments/assets/835d6405-1a3b-4142-bdce-075f0dfc79ee)

```python
df.info()
```

![image](https://github.com/user-attachments/assets/e1fa3d62-eb55-4807-a5f0-b9ee8e3745f2)

```python
df_null_sum=df.isnull().sum()
df_null_sum
```

![image](https://github.com/user-attachments/assets/3125286f-aee6-4a83-84c4-40f58a40df5a)

```python
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

```

![image](https://github.com/user-attachments/assets/7203335d-f7de-4221-a740-9faeedb50994)

```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

```

![image](https://github.com/user-attachments/assets/6442d99e-200a-4b98-bfeb-3f114b848aac)

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/13f6423a-f989-4ae9-8a70-73ec0788ee89)
```python
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/05260d87-297b-41d1-bf5f-bf20decfeb24)

```python

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/f9baa88b-550c-44b0-a5dc-485aba020bb6)



```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/0736711e-7d3b-4255-9c69-c9d9992d8f2b)

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/81b15fe8-424c-4892-94e9-b84e758bdcdf)

```python
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/76f9bdc5-6c82-4396-97ce-8c6b6077dd4d)

```python
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/d5b425e9-5815-488d-a34b-8cb65b93ea1e)

```python
# @title
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/1206aa4c-41f8-4c82-9ff3-f927cd993499)

```python
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/56fbe07f-c91c-4d2c-81bc-1dad9c452403)

```python
# @title
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/5f580439-9d06-4917-9a64-971eed2a2fa6)

```python
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif # Importing SelectKBest and f_classif
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
print("\nSelected features using ANOVA:")
print(selected_features_anova)

```
![image](https://github.com/user-attachments/assets/dcb08775-152e-48d3-9e85-1179a002e2c5)


```python
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/56305087-89e7-4280-b0ec-0bda7a5710bd)

```python
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/ce0f322b-0f2a-4ff1-bb9f-fcbe46f798d0)

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/d94455eb-a39a-4958-b99a-4902b81d8f67)

```python
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/a875d33c-bb66-4fea-b63c-8db250ab99a1)

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/de124ab9-f99b-4d34-8467-a9c4583c75e2)

# RESULT
THUS , Feature selection and feature scaling has been used on the given dataset
