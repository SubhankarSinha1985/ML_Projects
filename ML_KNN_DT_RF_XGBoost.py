# -*- coding: utf-8 -*-
"""
This python code has four ml classifiers - KNN, DecisionTree, Random Forest and XGboost
@author: subha
"""

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,confusion_matrix,recall_score,precision_score,f1_score,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


# importing dataset
data = pd.read_csv('C:\\Users\\subha\\Downloads\\SEMESTER - 2\\7072-MachineLearning\\Assignment_freshStart\\online_shoppers_intention.csv')
print(data.info())
print(data.shape)

# checking the presence of duplicate records
print(data.duplicated)
# dropping the duplicate records
data.drop_duplicates(inplace=True)
print(data.shape)
# checking the percentage of null values
print(data.isnull().mean()*100)

# checking class imbalance
print(data['Revenue'].value_counts())
# plotting the classes to visualise the imbalance
plt.figure()
sns.countplot(x='Revenue', data = data)
plt.show()

# checking correlations between numerical features
data_numerical_cols = data.drop(columns=['Month','VisitorType','Weekend','Revenue'])
print(type(data_numerical_cols ))
plt.figure(figsize = (10,8))
plt.rcParams.update({'font.size': 12})
sns.heatmap(data_numerical_cols.corr(), vmin = -1, vmax = 1, center = 0, annot=True, fmt=".2f", square=True, linewidths=.5)
plt.show()

# dropping highly correlated columns
data_col_dropped = data.drop(columns=['ProductRelated_Duration', 'BounceRates'])
print(data_col_dropped.shape)

#separating features from label/ target column from the dataset
features = data_col_dropped.drop(columns='Revenue')
target = data_col_dropped.iloc[:,-1]

# Performing EDA
# checking outliers by plotting boxplot of all the numeric columns
plt.title('Administrative')
sns.boxplot(features['Administrative'])
plt.title('Administrative_Duration')
sns.boxplot(features['Administrative_Duration'])
plt.title('Informational')
sns.boxplot(features['Informational'])
plt.title('Informational_Duration')
sns.boxplot(features['Informational_Duration'])
plt.title('ProductRelated')
sns.boxplot(features['ProductRelated'])
plt.title('ExitRates')
sns.boxplot(features['ExitRates'])
plt.title('PageValues')
sns.boxplot(features['PageValues'])
plt.title('SpecialDay')
sns.boxplot(features['SpecialDay'])
plt.title('OperatingSystems')
sns.boxplot(features['OperatingSystems'])
plt.title('Browser')
sns.boxplot(features['Browser'])
plt.title('Region')
sns.boxplot(features['Region'])
plt.title('TrafficType')
sns.boxplot(features['TrafficType'])

# Checking the distributions of all the numeric columns by plotting density curve
sns.displot(features['Administrative'],kind='kde')
sns.displot(features['Administrative_Duration'],kind='kde')
sns.displot(features['Informational'],kind='kde')
sns.displot(features['Informational_Duration'],kind='kde')
sns.displot(features['ProductRelated'],kind='kde')
sns.displot(features['ExitRates'],kind='kde')
sns.displot(features['PageValues'],kind='kde')
sns.displot(features['SpecialDay'],kind='kde')
sns.displot(features['OperatingSystems'], kind='kde')
sns.displot(features['Browser'],kind='kde')
sns.displot(features['Region'],kind='kde')
sns.displot(features['TrafficType'],kind='kde')

#DATA preprocessing 
# capping outliers with the upper and lower values
# Administrative column
q1_Administrative = features['Administrative'].quantile(0.25)
q3_Administrative = features['Administrative'].quantile(0.75)
IQR_Administrative = q3_Administrative - q1_Administrative
lower_Administrative = q1_Administrative - 1.5*IQR_Administrative
upper_Administrative = q3_Administrative + 1.5*IQR_Administrative

features['Administrative'] = np.where(features['Administrative']>upper_Administrative,upper_Administrative,
                                     np.where(features['Administrative']<lower_Administrative, lower_Administrative,features['Administrative']))
sns.boxplot(features['Administrative'])
sns.displot(features['Administrative'],kind='kde')

# Administrative_Duration column
q1_Administrative_Duration = features['Administrative_Duration'].quantile(0.25)
q3_Administrative_Duration = features['Administrative_Duration'].quantile(0.75)
IQR_Administrative_Duration = q3_Administrative_Duration - q1_Administrative_Duration
lower_Administrative_Duration = q1_Administrative_Duration - 1.5*IQR_Administrative_Duration
upper_Administrative_Duration = q3_Administrative_Duration + 1.5*IQR_Administrative_Duration

features['Administrative_Duration'] = np.where(features['Administrative_Duration']>upper_Administrative_Duration,upper_Administrative_Duration,
                                     np.where(features['Administrative_Duration']<lower_Administrative_Duration,lower_Administrative_Duration,features['Administrative_Duration']))

sns.boxplot(features['Administrative_Duration'])
sns.displot(features['Administrative_Duration'],kind='kde')

# ProductRelated column
print(features['ProductRelated'].describe())
q1_ProductRelated = features['ProductRelated'].quantile(0.25)
q3_ProductRelated = features['ProductRelated'].quantile(0.75)
IQR_ProductRelated = q3_ProductRelated - q1_ProductRelated
lower_ProductRelated = q1_ProductRelated - 1.5*IQR_ProductRelated
upper_ProductRelated = q3_ProductRelated + 1.5*IQR_ProductRelated

features['ProductRelated'] = np.where(features['ProductRelated']>upper_ProductRelated,upper_ProductRelated,
                                     np.where(features['ProductRelated']<lower_ProductRelated,lower_ProductRelated,features['ProductRelated']))

sns.boxplot(features['ProductRelated'])
sns.displot(features['ProductRelated'],kind='kde')

# ExitRates column
print(features['ExitRates'].describe())
q1_ExitRates = features['ExitRates'].quantile(0.25)
q3_ExitRates = features['ExitRates'].quantile(0.75)
IQR_ExitRates = q3_ExitRates - q1_ExitRates
lower_ExitRates = q1_ExitRates - 1.5*IQR_ExitRates
upper_ExitRates = q3_ExitRates + 1.5*IQR_ExitRates

features['ExitRates'] = np.where(features['ExitRates']>upper_ExitRates,upper_ExitRates,
                                     np.where(features['ExitRates']<lower_ExitRates,lower_ExitRates,features['ExitRates']))
sns.boxplot(features['ExitRates'])
sns.displot(features['ExitRates'],kind='kde')

# OperatingSystems column

q1_OperatingSystems = features['OperatingSystems'].quantile(0.25)
q3_OperatingSystems = features['OperatingSystems'].quantile(0.75)
IQR_OperatingSystems = q3_OperatingSystems - q1_OperatingSystems
lower_OperatingSystems = q1_OperatingSystems - 1.5*IQR_OperatingSystems
upper_OperatingSystems = q3_OperatingSystems + 1.5*IQR_OperatingSystems

features['OperatingSystems'] = np.where(features['OperatingSystems']>upper_OperatingSystems,upper_OperatingSystems,
                                     np.where(features['OperatingSystems']<lower_OperatingSystems, lower_OperatingSystems,features['OperatingSystems']))
sns.boxplot(features['OperatingSystems'])
sns.displot(features['OperatingSystems'],kind='kde')

# Region column
print(features['Region'].describe())
q1_Region = features['Region'].quantile(0.25)
q3_Region = features['Region'].quantile(0.75)
IQR_Region = q3_Region - q1_Region
lower_Region = q1_Region - 1.5*IQR_Region
upper_Region = q3_Region + 1.5*IQR_Region
features['Region'] = np.where(features['Region']>upper_Region,upper_Region,
                                     np.where(features['Region']<lower_Region,lower_Region,features['Region']))
sns.boxplot(features['Region'])
sns.displot(features['Region'],kind='kde')

# TrafficType column
print(features['TrafficType'].describe())
q1_TrafficType = features['TrafficType'].quantile(0.25)
q3_TrafficType = features['TrafficType'].quantile(0.75)
IQR_TrafficType = q3_TrafficType - q1_TrafficType
lower_TrafficType = q1_TrafficType - 1.5*IQR_TrafficType
upper_TrafficType = q3_TrafficType + 1.5*IQR_TrafficType
features['TrafficType'] = np.where(features['TrafficType']>upper_TrafficType,upper_TrafficType,
                                     np.where(features['TrafficType']<lower_TrafficType,lower_TrafficType,features['TrafficType']))
sns.boxplot(features['TrafficType'])
sns.displot(features['TrafficType'],kind='kde')


# Following features - 'Informational','Informational_Duration', 'PageValues', 'SpecialDay', 'Browser are windsorised as they have min, 25, 50, 75 percentile values are only 0
# windosorisation by top and bottom 5% to avoid the effect of outliers(https://www.geeksforgeeks.org/winsorization/)
print(features['Informational'].describe())
features['Informational'] = winsorize(features['Informational'],(0.05,0.05))
sns.displot(features['Informational'],kind='kde')

print(features['Informational_Duration'].describe())
features['Informational_Duration'] = winsorize(features['Informational_Duration'],(0.05,0.05))
sns.boxplot(features['Informational_Duration'])
sns.displot(features['Informational_Duration'],kind='kde')

print(features['PageValues'].describe())
features['PageValues'] = winsorize(features['PageValues'],(0.05,0.05))
sns.displot(features['PageValues'],kind='kde')

print(features['Browser'].describe())
features['Browser'] = winsorize(features['Browser'],(0.05,0.05))
sns.displot(features['SpecialDay'],kind='kde')

print(features['SpecialDay'].describe())
features['SpecialDay'] = winsorize(features['SpecialDay'],(0.05,0.05))
sns.displot(features['SpecialDay'],kind='kde')


# splitting train and test
xtrain, xtest = train_test_split(features, test_size=0.20)
print(xtrain.info())
print(xtrain.info())
ytrain, ytest = train_test_split(target, test_size=0.20)

# column transformation with scaling and One hot encoding
scaler = StandardScaler()
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ct = ColumnTransformer(transformers=[('OHE', ohe, [8,13,14]),
                                     ('Scale',scaler,[0,1,2,3,4,5,6,7,9,10,11,12])],
                       remainder='passthrough',
                       n_jobs=-1)

xtrain_transformed = ct.fit_transform(xtrain)
xtest_transformed = ct.transform(xtest)

# label encoding for target column
le = LabelEncoder()
ytrain_encoded = le.fit_transform(ytrain)
ytest_encoded = le.transform(ytest)

count_1 = 0
count_0 = 0

for i in range(len(ytrain_encoded)):
        if ytrain_encoded[i] == 1:
           count_1 = count_1 + 1
        else:
            count_0 = count_0 + 1
           
print(count_1)
print(count_0)

# Balancing the classes with SMOTE technique
sm = SMOTE(k_neighbors=5, random_state=42)
xtrain_res, ytrain_res = sm.fit_resample(xtrain_transformed, ytrain_encoded)

# checking the count of classes after balancing
count_1 = 0
count_0 = 0

for i in range(len(ytrain_res)):
        if ytrain_res[i] == 1:
           count_1 = count_1 + 1
        else:
            count_0 = count_0 + 1
           
print(count_1)
print(count_0)

# Balancing the classes with SMOTE + Tomek technique
smtomek = SMOTETomek(random_state=120)
xtrain_tomek, ytrain_tomek = smtomek.fit_resample(xtrain_transformed, ytrain_encoded)
print(xtrain_tomek.shape)

count_1 = 0
count_0 = 0

for i in range(len(ytrain_tomek)):
        if ytrain_tomek[i] == 1:
           count_1 = count_1 + 1
        else:
            count_0 = count_0 + 1
           
print(count_1)
print(count_0)

# Training and testing model - KNN
knn_clf = KNeighborsClassifier()
# 10 fold cross validation on Training data
cv_score = cross_val_score(knn_clf, xtrain_res, ytrain_res, cv=5)
print(cv_score.mean())
#training the model
model_knn = knn_clf.fit(xtrain_res,ytrain_res)
# predicting the model
ypred_knn = model_knn.predict(xtest_transformed)
# getting Accuracy score of the model
score_knn = accuracy_score(ytest_encoded, ypred_knn)
print('Accuracy Score of KNN',score_knn)
cm = confusion_matrix(ytest_encoded, ypred_knn)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=knn_clf.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)


# Training KNN model after SMOTE + TOMEK over sampling
knn_clf = KNeighborsClassifier()
cv_score = cross_val_score(knn_clf, xtrain_tomek, ytrain_tomek, cv=5)
print(cv_score.mean())
model_knn = knn_clf.fit(xtrain_tomek,ytrain_tomek)
ypred_knn = model_knn.predict(xtest_transformed)
score_knn = accuracy_score(ytest_encoded, ypred_knn)

print('Accuracy Score of KNN',score_knn)
cm = confusion_matrix(ytest_encoded, ypred_knn)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=knn_clf.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)


# Hyper-parameter tuning with 10 fold cross validation
params = [{'n_neighbors': [3,5,7,9,11,13],
          'weights':['uniform','distance']
         }]
gcv = GridSearchCV(KNeighborsClassifier(), param_grid=params, scoring='accuracy', cv=5,n_jobs=-1)
gcv.fit(xtrain_res, ytrain_res)
# best parameters
print(gcv.best_params_)
# best score
print(gcv.best_score_)


# Training KNN classifier with best parameters
knn_best = KNeighborsClassifier(n_neighbors=3,weights='distance')
# traing the model
model_knn = knn_best.fit(xtrain_res,ytrain_res)
# Predicting with test data
ypred_knn = model_knn.predict(xtest_transformed)
score_knn = accuracy_score(ytest_encoded, ypred_knn)
print(score_knn)

print('Accuracy Score of KNN',score_knn)
cm = confusion_matrix(ytest_encoded, ypred_knn)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=knn_best.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)


# PCA on data resampled after SMOTE method
print(xtrain_res.shape[1])
pca = PCA()
pca.fit_transform(xtrain_res)
plt.figure()
plt.grid(True)
plt.title('Principal Component Analysis')
plt.xlabel('Number of components')
plt.ylabel('Explained Varience')
plt.plot(range(xtrain_res.shape[1]), np.cumsum(pca.explained_variance_ratio_))
plt.show()

plt.figure()
plt.grid(True)
plt.title('Principal Component Analysis')
plt.xlabel('Number of components')
plt.ylabel('Explained Varience')
plt.bar(range(xtrain_res.shape[1]), pca.explained_variance_ratio_)
plt.show()

pca = PCA(n_components=9)
xtrain_pca = pca.fit_transform(xtrain_res)
xtest_pca = pca.transform(xtest_transformed)


# Training KNN classifier with best parameters with data resampled after SMOTE + Tomek method
knn_best_tomek = KNeighborsClassifier(n_neighbors=3,weights='distance')
# traing the model
model_knn_tomek = knn_best_tomek.fit(xtrain_tomek, ytrain_tomek)
ypred_knn = model_knn_tomek.predict(xtest_transformed)
score_knn = accuracy_score(ytest_encoded, ypred_knn)

print('Accuracy Score of KNN',score_knn)
cm = confusion_matrix(ytest_encoded, ypred_knn)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=knn_best_tomek.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)




knn_best_pca = KNeighborsClassifier(n_neighbors=3,weights='distance')
# traing the model
model_knn_pca = knn_best_pca.fit(xtrain_pca, ytrain_res)
ypred_knn_pca = model_knn_pca.predict(xtest_pca)
score_knn_pca = accuracy_score(ytest_encoded, ypred_knn_pca)
print('After PCA',score_knn_pca)

print('Accuracy Score of KNN',score_knn_pca)
cm = confusion_matrix(ytest_encoded, ypred_knn_pca)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=knn_best_pca.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)



# predicting probability
y_pred_proba = model_knn_pca.predict_proba(xtest_pca)[:,1]
print(y_pred_proba)

# Generating AUC-ROC-Curve for KNN
KNN_FPR,KNN_TPR,KNN_threshold = roc_curve(ytest_encoded, y_pred_proba)
KNN_acu = roc_auc_score(ytest_encoded, y_pred_proba)
print(KNN_acu)

# plotting ROC-AUC Curve for KNN
plt.figure()
plt.title('ROC-AUC Curve of KNN')
plt.xlabel('FPR - False Positive Rate')
plt.ylabel('TPR - True Positive Rate')
plt.plot(KNN_FPR,KNN_TPR, label="auc="+ str(KNN_acu))
plt.plot([0, 1], [0, 1],'r--')
plt.legend()
plt.show

###### Decision Tree ########

# Training the Decision Tree model with the balanced data by SMOTE + Tomek method
dt_clf = DecisionTreeClassifier()
# Doing 5-Fold cross validation on training data
cv_score = cross_val_score(dt_clf, xtrain_res, ytrain_res, cv=5)
print(cv_score.mean())
model_dt = dt_clf.fit(xtrain_tomek,ytrain_tomek)
ypred_dt = model_dt.predict(xtest_transformed)
score_dt = accuracy_score(ytest_encoded, ypred_dt)
print('Accuracy Score of DT',score_dt)
cm = confusion_matrix(ytest_encoded, ypred_dt)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=dt_clf.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)

# using GridsearchCV for hyper parameter tuning with 5 fold cross validation
params = [{'max_depth': [5,10,20,25,30,40,50],
           'criterion':['gini','entropy','log_loss'],
          'splitter':['random','best'],
          'min_samples_split':[5,10,15,20],
          
         }]

gcv = GridSearchCV(DecisionTreeClassifier(), param_grid=params, scoring='accuracy', cv=5,n_jobs=-1)
gcv.fit(xtrain_res, ytrain_res)
print(gcv.best_params_)
print(gcv.best_score_)

# Retraining the Decision Tree model with best parameters on the data balanced with SMOTE method
dt_best = DecisionTreeClassifier(max_depth=50,splitter='random',min_samples_split=5)
model_dt_best = dt_best.fit(xtrain_res,ytrain_res)
ypred_dt_best = model_dt_best.predict(xtest_transformed)
score_dt_best = accuracy_score(ytest_encoded, ypred_dt_best)

print('Accuracy Score of DT',score_dt_best)
cm = confusion_matrix(ytest_encoded, ypred_dt_best)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=dt_best.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)



# Retraining the Decision Tree model with best parameters on the data balanced with SMOTE+Tomek method
dt_best_tomek = DecisionTreeClassifier(max_depth=50,splitter='random',min_samples_split=5)
model_dt_best_tomek = dt_best_tomek.fit(xtrain_tomek,ytrain_tomek)
ypred_dt_best_tomek = model_dt_best_tomek.predict(xtest_transformed)
score_dt_best_tomek = accuracy_score(ytest_encoded, ypred_dt_best_tomek)

print('Accuracy Score of DT',score_dt_best_tomek)
cm = confusion_matrix(ytest_encoded, ypred_dt_best_tomek)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=dt_best_tomek.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)

# Training the model after PCA was applied to dataset

dt_best_pca = DecisionTreeClassifier(max_depth=50,splitter='random',min_samples_split=5)
model_dt_best_pca = dt_best_pca.fit(xtrain_pca,ytrain_res)
ypred_dt_best_pca = model_dt_best_pca.predict(xtest_pca)
score_dt_best_pca = accuracy_score(ytest_encoded, ypred_dt_best_pca)
print('Accuracy Score of DT',score_dt_best_pca)
cm = confusion_matrix(ytest_encoded, ypred_dt_best_pca)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=dt_best_pca.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)

# predicting probability on data balanced with SMOTE method
y_pred_proba = model_dt_best_pca.predict_proba(xtest_pca)[:,1]
print(y_pred_proba)

# Generating AUC-ROC-Curve for Decision Tree
DT_FPR, DT_TPR, DT_threshold = roc_curve(ytest_encoded, y_pred_proba)
DT_acu = roc_auc_score(ytest_encoded, y_pred_proba)
print(DT_acu)

# plotting ROC-AUC Curve for Decision Tree
plt.figure()
plt.title('ROC-AUC Curve of Decision Tree')
plt.xlabel('FPR - False Positive Rate')
plt.ylabel('TPR - True Positive Rate')
plt.plot(DT_FPR,DT_TPR, label="auc="+ str(DT_acu))
plt.plot([0, 1], [0, 1],'r--')
plt.legend()
plt.show

######## Training the Random Forest Algorithm #####

# Training the Random Forest model with the balanced data by SMOTE method
rf_clf = RandomForestClassifier()
cv_score = cross_val_score(rf_clf, xtrain_res, ytrain_res,  cv=5)
print(cv_score.mean())
model_rf = rf_clf.fit(xtrain_res,ytrain_res)
ypred_rf = model_rf.predict(xtest_transformed)
score_rf = accuracy_score(ytest_encoded, ypred_rf)
print(score_rf)
print('Accuracy Score of RF',score_rf)
cm = confusion_matrix(ytest_encoded, ypred_rf)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf_clf.classes_)
cm_disp.plot()
plt.show()
TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)

# Training the Random Forest model with the balanced data by SMOTE + Tomek method
rf_clf = RandomForestClassifier()
cv_score = cross_val_score(rf_clf, xtrain_tomek, ytrain_tomek,  cv=5)
print(cv_score.mean())
model_rf = rf_clf.fit(xtrain_tomek,ytrain_tomek)
ypred_rf = model_rf.predict(xtest_transformed)
score_rf = accuracy_score(ytest_encoded, ypred_rf)
print('Accuracy Score of RF',score_rf)
cm = confusion_matrix(ytest_encoded, ypred_rf)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf_clf.classes_)
cm_disp.plot()
plt.show()
TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)

# using GridsearchCV for hyper parameter tuning with 5-Fold cross validation on training data
params = [{'n_estimators':[100,200,300,400],
           'criterion':['gini','entropy'],
           'max_depth':[10,15,20,25]
         }]
gcv = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring='accuracy', cv=5,n_jobs=-1)
gcv.fit(xtrain_res, ytrain_res)
print(gcv.best_params_)
print(gcv.best_score_)

# Retraining the Random forest model with best parameters on the data balanced with SMOTE method
rf_best = RandomForestClassifier(n_estimators=300, max_depth=25, n_jobs=-1,)
model_rf_best = rf_best.fit(xtrain_res,ytrain_res)
ypred_rf_best = model_rf_best.predict(xtest_transformed)
score_rf_best = accuracy_score(ytest_encoded, ypred_rf_best)
print('Accuracy Score of RF',score_rf_best)
cm = confusion_matrix(ytest_encoded, ypred_rf_best)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf_best.classes_)
cm_disp.plot()
plt.show()
TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)


# Retraining the Random Forest model with best parameters on the data balanced with SMOTE+Tomek method
rf_best_tomek = RandomForestClassifier(n_estimators=300, max_depth=25, n_jobs=-1,)
model_rf_best_tomek = rf_best_tomek.fit(xtrain_tomek,ytrain_tomek)
ypred_rf_best_tomek = model_rf_best_tomek.predict(xtest_transformed)
score_rf_best_tomek = accuracy_score(ytest_encoded, ypred_rf_best_tomek)
print('Accuracy Score of RF',score_rf_best_tomek)
cm = confusion_matrix(ytest_encoded, ypred_rf_best_tomek)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf_best_tomek.classes_)
cm_disp.plot()
plt.show()
TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)


# Retraining RF classifier after PCA
rf_best_pca = RandomForestClassifier(n_estimators=300, max_depth=25, n_jobs=-1,)
model_rf_best_pca = rf_best_pca.fit(xtrain_pca,ytrain_res)
ypred_rf_best_pca = model_rf_best_pca.predict(xtest_pca)
score_rf_best_pca = accuracy_score(ytest_encoded, ypred_rf_best_pca)
print('Accuracy Score of RF',score_rf_best_pca)
cm = confusion_matrix(ytest_encoded, ypred_rf_best_pca)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf_best_pca.classes_)
cm_disp.plot()
plt.show()
TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)


# predicting probability for ROC for SMOTE balanaced data
y_pred_proba = model_rf_best_pca.predict_proba(xtest_pca)[:,1]
print(y_pred_proba)

# Generating AUC-ROC-Curve for Random Forest
RF_FPR, RF_TPR, RF_threshold = roc_curve(ytest_encoded, y_pred_proba)
RF_acu = roc_auc_score(ytest_encoded, y_pred_proba)
print(RF_acu)

# plotting ROC-AUC Curve for Random Forest
plt.figure()
plt.title('ROC-AUC Curve of Random Forest')
plt.xlabel('FPR - False Positive Rate')
plt.ylabel('TPR - True Positive Rate')
plt.plot(RF_FPR,RF_TPR, label="auc="+ str(RF_acu))
plt.plot([0, 1], [0, 1],'r--')
plt.legend()
plt.show

############## XGBoost ###########
# creating model with XGBclassifier
xgb = XGBClassifier(learning_rate=0.01,
                    n_estimators = 300,
                    max_depth=6,
                    min_child_weight=6,
                    gamma=0,
                    subsample=0.8,
                    nthread=4,
                    n_jobs=4,
                    seed = 80)
model = xgb.fit(xtrain_pca, ytrain_res)
y_pred_xgb = model.predict(xtest_pca)
accuracy_xgb = accuracy_score(ytest_encoded, y_pred_xgb)
print('accuracy score of XGBoost', accuracy_xgb)
cm = confusion_matrix(ytest_encoded, y_pred_xgb)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=xgb.classes_)
cm_disp.plot()
plt.show()

TruePositive = cm[1,1]
FalseNegative = cm[1,0]
Sensitivity = TruePositive / (TruePositive + FalseNegative)
print('Sensitivity',Sensitivity)
TrueNegative = cm[0,0]
FalsePositive = cm[0,1]
Specificity = TrueNegative / (FalsePositive + TrueNegative)
print('Specificity',Specificity)
geometric_mean = np.sqrt(Sensitivity*Specificity)
print('geometric mean',geometric_mean)

# predicting probability for ROC for SMOTE balanaced data
y_pred_proba = xgb.predict_proba(xtest_pca)[:,1]
print(y_pred_proba)

# Generating AUC-ROC-Curve for XGBoost
RF_FPR, RF_TPR, RF_threshold = roc_curve(ytest_encoded, y_pred_proba)
RF_acu = roc_auc_score(ytest_encoded, y_pred_proba)
print(RF_acu)

# plotting ROC-AUC Curve for XGBoost
plt.figure()
plt.title('ROC-AUC Curve of XGBoost')
plt.xlabel('FPR - False Positive Rate')
plt.ylabel('TPR - True Positive Rate')
plt.plot(RF_FPR,RF_TPR, label="auc="+ str(RF_acu))
plt.plot([0, 1], [0, 1],'r--')
plt.legend()
plt.show