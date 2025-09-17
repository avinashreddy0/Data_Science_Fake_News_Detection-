import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble  import IsolationForest
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,precision_score
import joblib


#loading data set using pandas 

df = pd.read_csv(r"C:\Users\indur\Downloads\fake_news_dataset.csv")
print(df)

print('sucessfully we load daat using pandas-----------------------------------')

#exploroed data analysis(EDA)
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())

print('we show exploarated data analysis----------------------------------------------')

# missing valued and data cleaning 

df = df.dropna()
df = df.drop_duplicates()
df['Title'] = df['Title'].str.title()
df['Description'] = df['Description'].str.title()
df['Label'] = df['Label'].str.title()

print('we cleaned data sucessfully-----------------------------------------------')

#transforming data into numeric from string---

df['Titles'] = df['Title'] +' '+ df['Description']

vectorizer = TfidfVectorizer(max_features=5000)
x = vectorizer.fit_transform(df['Titles']) 

joblib.dump(vectorizer,'for transform.pkl')

print('we transfor this columns in to numeric from string -----------------tfidvectorizer')

#labelencoder

label = LabelEncoder()
y = label.fit_transform(df['Label'])
joblib.dump(label,'topredict.pkl')
print('we transfor this columns in to numeric from string ----------------- using labelencoder')



#train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)

#model 

model = GradientBoostingClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)

y_pred_labes = label.inverse_transform(y_pred)
print(y_pred_labes)

y_test_label = label.inverse_transform(y_test)
print(y_test_label)

#actual vs y_predict 

result = pd.DataFrame({
    'Actual':y_test_label,
    'y_predict':y_pred_labes
})

print(result)

#metics :

print('accuracy_score:',accuracy_score(y_test_label,y_pred_labes))
print('confusction_metics:',confusion_matrix(y_test_label,y_pred_labes))

#saving in joblib

joblib.dump(model,'fack_new_dectorer.pkl')