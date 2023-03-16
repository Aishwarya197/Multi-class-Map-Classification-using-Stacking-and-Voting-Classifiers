#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from imblearn.pipeline import Pipeline

train_df = pd.read_csv("train.csv")


# In[2]:


#To prevent warnings being displayed
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore") #ignoring the warnings


# In[3]:


train_df.shape


# In[4]:


train_df.head()


# In[5]:


train_df = train_df[["unique_id","class"]]


# In[6]:


train_df.head()


# In[7]:


data_df = pd.DataFrame()
for index, row in train_df.iterrows():
    temp_df = pd.read_csv("data-10/{}.csv".format(row['unique_id']),header=None)
    data_df = pd.concat([data_df, temp_df])


# In[8]:


data_df = data_df.reset_index(drop=True)


# In[9]:


data_df.shape


# In[10]:


data_df


# In[11]:


columns = ["feature_{}".format(str(i)) for i in range(1,1025)]
data_df.columns = columns


# In[12]:


data_df


# In[13]:


data_df["unique_id"] = train_df["unique_id"]
data_df["class"] = train_df["class"]
col_list = ["unique_id"]
col_list.extend(columns)
col_list.append("class")
data_df = data_df[col_list]
data_df


# In[14]:


data_df.to_csv("Mapped_train_data.csv",index=False)


# In[15]:


data_df = data_df.set_index("unique_id")
data_df


# In[16]:


data_df["class"].value_counts()


# In[17]:


#Analyzing the distribution of Class Label / Dependent Variable to choose the right metric for classification task
feature = "class"
feature_value_counts = data_df[feature].value_counts() #getting value counts for each unique value present in the attack column
feature_value_counts = pd.DataFrame({feature:feature_value_counts.index,'Value Count':feature_value_counts.values}) #Converting the series into Dataframe object
plt.figure(figsize=(20,10))
s = sns.barplot(x = feature, y="Value Count",data=feature_value_counts) # Plotting the value counts of each unique value present in the attack
s.set_xticklabels(s.get_xticklabels()) #setting x-ticks for the plot
s.set_title("Value Counts of the "+feature) #setting title for the plot
plt.show()
feature_percentages = data_df[feature].value_counts().reset_index(name ="counts") #getting value counts for each unique value present in the attack column
feature_percentages.rename(columns = {'index':feature}, inplace = True) #replacing index name with attack
feature_percentages["Percentage"] = feature_percentages["counts"].apply(lambda x:round(x*100/feature_percentages["counts"].sum(),5)) # Computing the percentage value
plt.figure(figsize=(20,10))
s = sns.barplot(x = feature, y="Percentage",data=feature_percentages) #Plotting the percentage values of each unique value present in the attack
s.set_title(feature+" values distribution in terms of Percentages") #setting title for the plot
s.set_xticklabels(s.get_xticklabels()) #setting x-ticks for the plot
plt.show(); #to show the plots
print(feature_value_counts) # To print the value counts
print(feature_percentages) # To print the percentages


# In[19]:


data_df.isnull().sum().sum() #Checking for null values


# In[20]:


#Function to analyze the numerical features in the dataset
def plot_numerical_features(feature,data_df=data_df):
    s = sns.FacetGrid(data_df, hue="class", size=5).map(sns.distplot, feature).add_legend(); #Plotting value counts of all the records present in the data
    s.fig.suptitle("Univariate analysis of the feature {}".format(feature)) #Setting title to the plot
    plt.show(); # To display the plot
    print("Box Plots to analyze the {} distribution".format(feature))
    s = sns.boxplot(x='class',y=feature, data=data_df) #Plotting the box plot for the data distribution of the feature
    s.set_title("Box Plot for the feature "+feature)
    plt.show();
    print(data_df[feature].describe()) # To print the min, max, 25th, 50th , 75th etc., statistical value for all the records
    print("="*100)


# In[21]:


for col in columns:
    print("Analysis of {}".format(col))
    plot_numerical_features(col)
    print("="*50)


# In[23]:


temp_data_df = data_df.copy()
print("Dataset shape before removing the outliers ",temp_data_df.shape)
for feature in columns: 
    q_99 = np.percentile(temp_data_df[feature],99.9)
    temp_data_df.drop(temp_data_df[temp_data_df[feature]>q_99].index, inplace = True)
print("Dataset shape after removing the outliers ",temp_data_df.shape)


# In[33]:


#temp_data_df = data_df.copy()
print("Dataset shape before removing the outliers ",temp_data_df.shape)
q_99 = np.percentile(temp_data_df["feature_225"],99.9)
temp_data_df.drop(temp_data_df[temp_data_df["feature_225"]>q_99].index, inplace = True)
print("Dataset shape after removing the outliers ",temp_data_df.shape)


# In[24]:


data_df_corr = data_df.corr()
data_df_corr


# In[25]:


#Feature Filtering - Removing Features which are highly correlated by choosing a threshold of 0.90
#https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/
correlated_features = set()
class_feature = len(data_df_corr.columns)-1
for i in range(len(data_df_corr.columns)):
    for j in range(i):
        if abs(data_df_corr.iloc[i, j]) > 0.90:
            print("Correlation value between features {} and {} is {}".format(data_df_corr.columns[i],data_df_corr.columns[j],data_df_corr.iloc[i, j]))
            if abs(data_df_corr.iloc[i,class_feature]) > abs(data_df_corr.iloc[j,class_feature]):
                print("Correlation value between feature {} and {} is {}".format(data_df_corr.columns[i],"class",data_df_corr.iloc[i, class_feature]))
                print("Correlation value between feature {} and {} is {}".format(data_df_corr.columns[j],"class",data_df_corr.iloc[j, class_feature]))
                print("{} is highly correlated with {} than feature {} hence removing {}".format(data_df_corr.columns[i],class_feature,data_df_corr.columns[j],data_df_corr.columns[j]))
                colname = data_df_corr.columns[j]
            else:
                print("Correlation value between feature {} and {} is {}".format(data_df_corr.columns[i],"class",data_df_corr.iloc[i, class_feature]))
                print("Correlation value between feature {} and {} is {}".format(data_df_corr.columns[j],"class",data_df_corr.iloc[j, class_feature]))
                print("{} is highly correlated with {} than feature {} hence removing {}".format(data_df_corr.columns[j],class_feature,data_df_corr.columns[i],data_df_corr.columns[i]))
                colname = data_df_corr.columns[i]
            correlated_features.add(colname)
            print("="*10)
print(correlated_features)


# In[26]:


data_df.drop(columns=['feature_211', 'feature_233', 'feature_107', 'feature_336'],inplace=True)


# In[27]:


data_df.shape


# In[28]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE,SequentialFeatureSelector
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix,f1_score,precision_score,recall_score,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
rs = 0


# In[29]:


#Splitting the dataset into train dataset and test dataset
data_df_features = data_df.drop(columns=["class"]) #Extracting Features
data_df_class = data_df[["class"]] #Saving only Class labels

"""Splitting the data to Train and Test in a proportion of 80% and 20% respectively, since the data is imbalanced we
want to reserve the proportion of the imbalance in train and test set hence we would be using startify parameter which
does the splitting based on the values distribution present in the class label column"""
X_train, X_test, y_train, y_test = train_test_split(data_df_features, data_df_class, test_size=0.20, random_state=rs, stratify = data_df_class)


# In[76]:


#Hyperparameter Tuning of Decision Tree Classifier
from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(random_state=rs)
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt, param_grid=params, cv=5, n_jobs=-1, verbose=1, scoring = "f1_macro")
grid_search.fit(X_train, y_train)


# In[77]:


#Fitting the Decision Tree model on Train data and predicting it on the test data
dt_best = grid_search.best_estimator_
print(dt_best)
dt_best.fit(X_train, y_train) #Fitting on the train data
dt_yPred = dt_best.predict(X_test) #Predicting on test data


# In[84]:


#Function to obtain different metrics based on predicted class labels
def print_metrics(y_test,y_pred,name):
    cm = confusion_matrix(y_test, y_pred) #Computing the Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") #Initializing the Heatmap
    plt.xlabel("Predicted Labels") #Setting x label
    plt.ylabel("Actual Labels") #Setting y label
    plt.title("Confusion Matrix Obtained") #Setting the title
    plt.show(); #Showing the plot
    f1_value = round(f1_score(y_test,y_pred,labels=np.unique(y_pred),average='macro'),4) #Calculating F1-score value
    prec = round(precision_score(y_test,y_pred,labels=np.unique(y_pred),average='macro'),4) #Calculating Precision value
    rec = round(recall_score(y_test,y_pred,labels=np.unique(y_pred),average='macro'),4) #Calculating Recall value
    print("F1-score obtained is ",f1_value) #Printing F1-score value
    print("Precision obtained is ",prec) #Printing Precision value
    print("Recall obtained is ",rec) #Printing Recall value
    print("Classification Report obtained is :")
    print(classification_report(y_test, y_pred))
    return [f1_value,prec,rec] #return f1-score, precision and recall


# In[85]:


results_table = []

#Performance Metrics obtained using Decision Tree
dt_list = ["Decision Tree"]
dt_list.extend(print_metrics(y_test, dt_yPred,"Decision Tree"))
results_table.append(dt_list)


# In[87]:


X_train.head()


# In[107]:


from tqdm import tqdm
from sklearn.decomposition import PCA

dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10,random_state=0)
f1_dict = {}

#for nf in tqdm(range(3,X_train.shape[1]+1)):
for nf in tqdm(range(3,min(X_train.shape[1],X_train.shape[0]))):
    f1_val = []
    pca = PCA(n_components=nf,random_state=rs)
    pca_train = pca.fit_transform(X_train)
    dt.fit(pca_train,y_train)
    pca_test = pca.transform(X_test)
    pca_test_predict = dt.predict(pca_test)
    f1_val.append(round(f1_score(y_test,pca_test_predict,labels=np.unique(pca_test_predict),average='macro'),4))
    f1_dict[nf] = round(sum(f1_val)/len(f1_val),4)
plt.figure(figsize=(12,10))
plt.plot(*zip(*sorted(f1_dict.items())),color='blue',marker='o')
Title = "Feature Selection with Dimensionality Reduction (PCA Method) for Decision Tree"
plt.title(Title, fontsize=16)
plt.xticks(range(0,min(X_train.shape[0],X_train.shape[1])+1,15))
plt.xlabel("Number of Features", fontsize=16)
plt.ylabel("f1_score", fontsize=16)
plt.show();
f1_adasyn_dt = max(f1_dict.values())
f1_nfea_adasyn_dt=  max(f1_dict, key=f1_dict.get)
print("DT Model - Maximum f1-score obtained using PCA is {} for the number of components = {}".format(f1_adasyn_dt,f1_nfea_adasyn_dt))


# In[113]:


dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, random_state=0)
pca = PCA(n_components=102,random_state=rs)
dt_pca_pipeline = Pipeline([('pca', pca), ('dt', dt)])
dt_pca_pipeline.fit(X_train,y_train)
dt_pca_pipeline_yPred = dt_pca_pipeline.predict(X_test)
dt_list = ["Decision Tree - After Hyperparameter Tuning and Feature Selection based on PCA"]
dt_list.extend(print_metrics(y_test, dt_pca_pipeline_yPred,"Decision Tree - PCA"))
results_table.append(dt_list)


# In[114]:


# def utility_select_feature(X_train, y_train, fsm, model):
#     f1_val = []
#     fs = fsm.fit(X_train, y_train)
#     Xtrain_new = fs.transform(X_train)
#     Xtest_new = fs.transform(X_test)
#     model.fit(Xtrain_new, y_train)
#     yPred = model.predict(Xtest_new)
#     f1_val.append(round(f1_score(y_test,yPred,labels=np.unique(yPred),average='macro'),4))
#     f1 = round(sum(f1_val)/len(f1_val),4)
#     return f1


# In[ ]:


# dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, random_state=0)
# acc_dict = {}
# f1_dict ={}
# for nf in tqdm(range(3,X_train.shape[1]+1)):
#     rfe = RFE(dt, n_features_to_select=nf)
#     f1_val = utility_select_feature(X_train, y_train, rfe, dt)
#     f1_dict[nf] = f1_val

# plt.figure(figsize=(12,10))
# plt.plot(*zip(*sorted(f1_dict.items())),color='blue',marker='o')
# Title = "Feature Selection with Recursive Feature Elimination for Decision Tree"
# plt.title(Title, fontsize=16)
# plt.xticks(range(0,processed_adasyn_train_data_trX.shape[1]+1))
# plt.xlabel("Number of Features", fontsize=16)
# plt.ylabel("f1_score", fontsize=16)
# plt.show();

# f1_adasyn_dt = max(f1_dict.values())
# f1_nfea_adasyn_dt=  max(f1_dict, key=f1_dict.get)
# print("DT Model - Maximum f1-score obtained using Recursive Feature Elimination Method {} for the number of features = {}".format(f1_adasyn_dt,f1_nfea_adasyn_dt))


# In[118]:


from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=rs)
processed_adasyn_data_df_features, processed_adasyn_data_df_class = adasyn.fit_resample(data_df_features, data_df_class)


# In[119]:


processed_adasyn_data_df_features.shape


# In[120]:


processed_adasyn_data_df_features


# In[121]:


processed_adasyn_data_df_class.value_counts()


# In[138]:


import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=791,random_state=rs)
pca.fit(processed_adasyn_data_df_features)

pca_values = pca.explained_variance_ratio_.cumsum()
pca_values_dict = {}
for index,pca in zip(range(1,pca_values.shape[0]),pca_values):
    pca_values_dict[index] = pca
    


# In[139]:


pca_values_dict


# In[152]:


print(max(pca_values_dict, key=pca_values_dict.get),max(pca_values_dict.values()))


# In[140]:


#Splitting the dataset into train dataset and test dataset

"""Splitting the data to Train and Test in a proportion of 80% and 20% respectively, since the data is imbalanced we
want to reserve the proportion of the imbalance in train and test set hence we would be using startify parameter which
does the splitting based on the values distribution present in the class label column"""
X_train, X_test, y_train, y_test = train_test_split(processed_adasyn_data_df_features, processed_adasyn_data_df_class, test_size=0.20, random_state=rs, stratify = processed_adasyn_data_df_class)


# In[141]:


#Hyperparameter Tuning of Decision Tree Classifier
from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(random_state=rs)
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt, param_grid=params, cv=5, n_jobs=-1, verbose=1, scoring = "f1_macro")
grid_search.fit(X_train, y_train)


# In[142]:


#Fitting the Decision Tree model on Train data and predicting it on the test data
dt_best = grid_search.best_estimator_
print(dt_best)
dt_best.fit(X_train, y_train) #Fitting on the train data
dt_yPred = dt_best.predict(X_test) #Predicting on test data


# In[143]:


#Performance Metrics obtained using Decision Tree
dt_list = ["Decision Tree - ADASYN"]
dt_list.extend(print_metrics(y_test, dt_yPred,"Decision Tree -ADASYN"))
results_table.append(dt_list)


# In[144]:


from tqdm import tqdm
from sklearn.decomposition import PCA

dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, random_state=0)
f1_dict = {}

#for nf in tqdm(range(3,X_train.shape[1]+1)):
for nf in tqdm(range(3,min(X_train.shape[1],X_train.shape[0]))+1):
    f1_val = []
    pca = PCA(n_components=nf,random_state=rs)
    pca_train = pca.fit_transform(X_train)
    dt.fit(pca_train,y_train)
    pca_test = pca.transform(X_test)
    pca_test_predict = dt.predict(pca_test)
    f1_val.append(round(f1_score(y_test,pca_test_predict,labels=np.unique(pca_test_predict),average='macro'),4))
    f1_dict[nf] = round(sum(f1_val)/len(f1_val),4)
plt.figure(figsize=(12,10))
plt.plot(*zip(*sorted(f1_dict.items())),color='blue',marker='o')
Title = "Feature Selection with Dimensionality Reduction (PCA Method) for Decision Tree"
plt.title(Title, fontsize=16)
plt.xticks(range(0,min(X_train.shape[0],X_train.shape[1])+1,15))
plt.xlabel("Number of Features", fontsize=16)
plt.ylabel("f1_score", fontsize=16)
plt.show();
f1_adasyn_dt = max(f1_dict.values())
f1_nfea_adasyn_dt=  max(f1_dict, key=f1_dict.get)
print("DT Model - Maximum f1-score obtained using PCA is {} for the number of components = {}".format(f1_adasyn_dt,f1_nfea_adasyn_dt))


# In[145]:


dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, random_state=0)
pca = PCA(n_components=151,random_state=rs)
dt_pca_pipeline = Pipeline([('pca', pca), ('dt', dt)])
dt_pca_pipeline.fit(X_train,y_train)
dt_pca_pipeline_yPred = dt_pca_pipeline.predict(X_test)
dt_list = ["Decision Tree - ADASYN After Hyperparameter Tuning and Feature Selection based on PCA"]
dt_list.extend(print_metrics(y_test, dt_pca_pipeline_yPred,"Decision Tree - ADASYN and PCA"))
results_table.append(dt_list)


# In[146]:


results_table


# In[148]:


#Tabulating the obtained results and displaying the results by formatting
results_df = pd.DataFrame(results_table,columns = ["Classifier","f1-score","Precision","Recall"])
with pd.option_context('display.precision', 4):
    formatted_table=(results_df.style.background_gradient(cmap ='RdYlGn'))
formatted_table


# In[ ]:




