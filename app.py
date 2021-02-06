import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from boruta import BorutaPy
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
#Train test split
from sklearn.model_selection import train_test_split

#Counter Imbalance data
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE

from kmodes.kmodes import KModes
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score

# Functions
#############################################################################
# Plot graph
def laundry_plot(df,by,y,stack=False,sort=False,kind='bar'):
    pivot = df.groupby(by)[y].count().unstack(y)
    pivot = pivot.sort_values(by=pivot.columns.to_list(),ascending=False)
    ax = pivot.plot(kind=kind,stacked=stack,figsize =(8,8))
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        
def laundry_heatmap(df,by,count_att,y):
    df = df.groupby(by)[count_att].count().unstack(y)
    fig = plt.figure(figsize=(5,5))
    sns.heatmap(df,annot =  df.values ,fmt='g')

        
#To add the number on graphs 
def auto_label():
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# To rank the  features of dataset
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

#############################################################################
st.title('Decision Support System')
st.title('Group Project')
st.title('Question 1 [Laundry Scenario]')
st.text('Mak Chin Ho 1151100769')
st.text('Dhevan A/L Rajandran 1161100744')


raw_df = pd.read_csv("LAUNDRY.csv")

st.subheader('Display Raw Data')
st.write(raw_df)

st.text('The date time is inconsistent, so join them')
st.text('Drop time and date fter combining and join the merged data into last column')
raw_df['Time'] = raw_df['Time'].str.replace(';',':')
raw_df['Date_Time'] =  pd.to_datetime(raw_df['Date'] + ' ' + raw_df['Time'], format='%d/%m/%Y %H:%M:%S')
raw_df['Date_Time'] 
raw_df.drop(['Date','Time'],axis='columns',inplace=True)
st.write(raw_df.head())

st.header('Data-Preprocessing')
st.subheader('Check Null Values')
st.write(raw_df.isnull().sum())

st.subheader('Check Unique Values')
st.write(raw_df.nunique())

st.header('Data Cleaning')
st.subheader('Imputing Null Value using KNN')

for_knn = raw_df.copy()

#for 2D array
encoder = OrdinalEncoder()
imputer = KNNImputer(n_neighbors=1)

#iterate through each column in the data
target = for_knn.columns[for_knn.isnull().sum()>0]
to_drop= for_knn.columns[for_knn.isnull().sum() == 0]
for_knn = for_knn.drop(to_drop,axis='columns')

for columns in target:
    nonulls = np.array(for_knn[columns].dropna())
    impute_reshape = nonulls.reshape(-1,1)
    impute_ordinal = encoder.fit_transform(impute_reshape)
    for_knn[columns].loc[for_knn[columns].notnull()] = np.squeeze(impute_ordinal)

for_knn_clean = pd.DataFrame(np.round(imputer.fit_transform(for_knn)), columns=target)

to_convert = [i for i in for_knn_clean if for_knn_clean[i].dtypes == 'float64']
for_knn_clean[to_convert] = for_knn_clean[to_convert].astype(int)
                   
to_convert = [i for i in for_knn_clean if for_knn_clean[i].dtypes == 'int']
for_knn_clean[to_convert] = for_knn_clean[to_convert].astype(str)
                   
for i in for_knn_clean.columns:
    res = np.sort(raw_df[i].dropna().unique())
    for count,j in enumerate(res):
        for_knn_clean[i] = for_knn_clean[i].replace(str(count), j)
for_knn_clean

for_knn_clean = for_knn_clean.merge(raw_df[to_drop],on=raw_df.index,left_index=True)
for_knn_clean = for_knn_clean.drop('key_0',axis='columns')

for_knn_clean.isnull().sum()

for_knn_clean

st.subheader('Handle Noisy Data')
st.text('To ensure no repeating values')
cleaned_df = for_knn_clean.copy()

for i,col in enumerate(cleaned_df.columns):
    print(col + ' : '  + str(cleaned_df[col].unique()))

st.write(cleaned_df)
st.text('Kids_Category and Pants_Colour have duplicated values, so we need to make them have unique values')
cleaned_df['Kids_Category'] = cleaned_df['Kids_Category'].replace(dict.fromkeys(['toddler','toddler '], 'toddler'))
cleaned_df['Pants_Colour'] = cleaned_df['Pants_Colour'].replace(dict.fromkeys(['blue','blue ','blue  ','blue_jeans'], 'blue'))
cleaned_df['Pants_Colour'] = cleaned_df['Pants_Colour'].replace(dict.fromkeys(['black','black '], 'black'))

for i,col in enumerate(cleaned_df.columns):
    print(col + ' : '  + str(cleaned_df[col].unique()))

st.write(cleaned_df)

st.header('Data Transformation')
st.subheader('Data Discretization')
st.text('XXX')

#
cleaned_df['Age_Range'] = cleaned_df['Age_Range'].astype(int)

age_bins = [i for i in range(cleaned_df['Age_Range'].min()-1,cleaned_df['Age_Range'].max()+1,4)]
age_label = ['28-31','32-35','36-39','40-43','44-47','48-51','52-55']

cleaned_df['Age_Bin'] = pd.cut(x= cleaned_df['Age_Range'], 
                                  bins = age_bins,
                                  labels = age_label
                                  )
cleaned_df['Age_Bin'] = cleaned_df['Age_Bin'].astype(str)
st.dataframe(cleaned_df[['Age_Bin']].head(10))

#
cleaned_df['Day'] = [i.day_name() for i in cleaned_df['Date_Time']]
print(cleaned_df['Day'].unique())
st.dataframe(cleaned_df[['Day']].head(10))

#
hour = []
for i in range(cleaned_df.shape[0]):
    hour.append(cleaned_df['Date_Time'].iloc[i].hour)

# print(cleaned_df.shape[0],len(hour))
# print("Hour range:",min(hour),max(hour)) #MAKE SURE ALL THE VALUES ARE WITHIN 0-23
# print('Bin: ', [i for i in range(-1,25,6)])
cleaned_df['Part_of_day'] = pd.cut(x=hour,
                                      bins=[i for i in range(-1,25,6)],
                                      labels=['Early Morning','Morning','Afternoon','Evening']
                                      )
# print(cleaned_df['Part_of_day'].unique())]
cleaned_df['Part_of_day'] = cleaned_df['Part_of_day'].astype(str)
st.write(cleaned_df[['Part_of_day']].head(10))


#display latest cleaned_df 
st.write(cleaned_df)

st.subheader('Differences before and after cleaning')
st.text('Raw dataframe')
st.dataframe(raw_df.info())

st.dataframe(cleaned_df.info())

st.write(raw_df.describe())
st.write(cleaned_df.describe())

st.subheader('Label Encoding')
st.text("Binary [ 'Gender', 'With_Kids', 'Spectacles' ]")
st.text("Ordinal [ 'Body_Size', 'Age_Bin','Day', 'Part_of_day', 'Basket_Size' ]")
st.text("Norminal [ 'Race', 'Kids_Category', 'Basket_colour', 'Attire', 'Shirt_Colour', 'Pants_Colour', 'shirt_type', 'pants_type', 'Wash_Item', 'Washer_No', 'Dryer_No' ]")

# Binary
binary = ['Gender','With_Kids','Spectacles']

# Ordinal
ordinal = ['Body_Size','Age_Bin','Basket_Size']
ordinal_ts = ['Day', 'Part_of_day']

# Norminal
norminal = ['Race','Kids_Category','Basket_colour','Attire','Shirt_Colour','Pants_Colour' ,'shirt_type','pants_type','Wash_Item','Washer_No', 'Dryer_No']

# print('Number of Binary columns                 : ', len(binary))
# print('Number of Ordinal columns                : ', len(ordinal))
# print('Number of Ordinal time series  columns   : ', len(ordinal_ts))
# print('Number of Norminal columns               : ', len(norminal))
# print()
# print('Total number of columns to be encoded    : ', len(ordinal)+len(norminal)+len(ordinal_ts)+len(binary))
#exclude = [date_time, Age_Range]
#exclude age_range because it is numerical data that don't need encoding

encoded_df = cleaned_df.copy()

#process the binary encode
binary_encoder = preprocessing.LabelBinarizer()
for i in binary:
    encoded_df[i]=binary_encoder.fit_transform(encoded_df[i])
    
#process for ordinal encoding
ordinal_encoder = OrdinalEncoder()
encoded_df[ordinal] = ordinal_encoder.fit_transform(encoded_df[ordinal])

#ordinal encoding day 
# Day: Monday=0, Tuesday=1 ... Sunday =6
# Part_of_day: Early Morning = 0, Afternoon =1 
label_ts = [['Monday', 'Tuesday' ,'Wednesday','Thursday','Friday','Saturday','Sunday'],
             ['Early Morning','Morning','Afternoon','Evening']]
ordinal_encoder_ts = OrdinalEncoder(label_ts)
encoded_df[ordinal_ts] = ordinal_encoder_ts.fit_transform(encoded_df[ordinal_ts])

#norminal encoding
norminal_encoder = preprocessing.LabelEncoder()
for i in norminal:
    encoded_df[i] = norminal_encoder.fit_transform(encoded_df[i])
    
to_convert = [i for i in encoded_df if encoded_df[i].dtypes == 'float64']
encoded_df[to_convert] = encoded_df[to_convert].astype(int)
encoded_df.dtypes
st.write(encoded_df)

st.header('Exploratory Data Analysis and Data Visualisation')
st.subheader('Exploratory Questions')
# Q1
st.subheader('1. Which hour is consider the daily peak hour(Most visited hour)?')
st.text('Based on the bar chart, the peak hour where the laundry is most visited is 4 am with 102 customers visited at the hour whereas the least visited hour is 5 am with only 3 visitors visited.')

laundry_hour = cleaned_df.copy()
laundry_hour['Hour'] = [i.hour for i in cleaned_df['Date_Time']]
#print(laundry_hour['Hour'].unique())
#laundry_hour[['Hour']].head(10)
hour=laundry_hour.groupby(["Hour"]).size().to_frame("Count")

plt.figure(figsize=(7,7))
ax = hour.plot(kind='bar')
auto_label()
plt.xticks(rotation=0)
st.bar_chart(hour)

# Q2
st.subheader('2. Which part of the day does customers like to visit? (According to gender)')
st.text('Although 4 am is the most visited hour but based on the bar chart below, it is shown that the laundry shop is often visited by customers during evening. Second most visited part of the day is early morning.')
laundry_plot(cleaned_df,['Part_of_day','Gender'],'Gender')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Q3
st.subheader('3. Which gender brings along their kids to the laundry shop?')
st.text('Based on the bar chart below, it is shown that female customers bring along their kids often compare to male customers.')
laundry_plot(cleaned_df,['Gender','With_Kids'],'With_Kids',stack=False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Q4
st.subheader('4. Customers of what age range tends to wear formal attire?')
st.text('Based on the bar chart below, it is shown that customers within age group of 44 to 47 often wear formal attire and casual attires are often wear by most customers.')
laundry_plot(cleaned_df,['Age_Bin','Attire'],'Attire')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

#Q5
st.subheader('5. What time do customers who wears formal attire visit laundry most?')
st.text('The bar chart below shows customers who often wash formal attire visit the laundry at early morning.')
laundry_plot(cleaned_df,['Part_of_day','Attire'],'Attire')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Q6
st.subheader('6. Which washer and dryer is most used to wash clothes?')
st.text('')
laundry_plot(cleaned_df,['Washer_No','Wash_Item'],'Wash_Item')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
laundry_plot(cleaned_df,['Dryer_No','Wash_Item'],'Wash_Item')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Q7
st.subheader('7. What is the relationship between shirt type and pants type?')
st.text('Based on the heatmap, most customers wear long pants and short sleeves followed by short pants and short sleeves. Customers who wears short pants and long sleeves are the least.')
shirt_pants = cleaned_df[['Wash_Item','shirt_type','pants_type']]
shirt_pants_all = shirt_pants.groupby(['shirt_type','pants_type'])['Wash_Item'].count().unstack('shirt_type')
plt.figure(figsize=(5,5))
heatmap = sns.heatmap(shirt_pants_all,annot = shirt_pants_all.values, fmt='g')
st.pyplot()

# Q8
st.subheader('8. Customers of which race often visits laundry?')
st.text('')
plt.figure(figsize=(7,7))
ax = cleaned_df['Race'].value_counts().plot(kind='bar')
auto_label()
plt.xticks(rotation=0)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Q9
st.subheader('9. Customers of which age group often visit laundry?')
st.text('')
plt.figure(figsize=(7,7))
ax = cleaned_df['Age_Bin'].value_counts().plot(kind='bar')
auto_label()
plt.xticks(rotation=0)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


st.subheader("The distribution of customers's age.")
st.text('')
ax = cleaned_df['Age_Range'].plot(kind='hist',bins=age_bins)
auto_label()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.subheader("The distribution of customers' age range and gender.")
st.text('')
age_gender = cleaned_df.groupby(['Age_Bin','Gender'])['Gender'].count().unstack('Gender')
age_gender = age_gender.sort_values(by=['female','male'],ascending=False)
age_gender.plot(kind='barh', stacked=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.header('Correlation Analysis')
st.text('Method of correlation analysis: \n - Pearson \n - Kendall \n - Spearman')

st.subheader('Pearson')
corr_matrix  = encoded_df.corr('pearson')
plt.figure(figsize=(30,30))
sns_heatmap = sns.heatmap(corr_matrix, vmax =.8,square = True,annot = True, fmt='.2f',annot_kws={'size':20} , cmap=sns.color_palette("Reds"))
st.pyplot()

st.subheader('Kendall')
corr_matrix  = encoded_df.corr('kendall')
plt.figure(figsize=(30,30))
sns_heatmap = sns.heatmap(corr_matrix, vmax =.8,square = True,annot = True, fmt='.2f',annot_kws={'size':20} , cmap=sns.color_palette("Reds"))
st.pyplot()


st.subheader('Spearman')
corr_matrix  = encoded_df.corr('spearman')
plt.figure(figsize=(30,30))
sns_heatmap = sns.heatmap(corr_matrix, vmax =.8,square = True,annot = True, fmt='.2f',annot_kws={'size':20} , cmap=sns.color_palette("Reds"))
st.pyplot()

st.header('Wash_Item_Weight calculation')
st.write(cleaned_df['Basket_Size'].value_counts())

adf = cleaned_df.copy()
adf['item_weight']=''

np.random.seed(6)
choice = ['heavy','light']

for i in (adf.index-1):
    if adf['Basket_Size'].get(i) == 'small':
        if adf['Day'].get(i) == 'Saturday' or adf['Day'].get(i) == 'Sunday':
            adf['item_weight'].iloc[i] = np.random.choice(choice,p=(0.5,0.5))
            
        else:
            adf['item_weight'].iloc[i] = np.random.choice(choice,p=(0.3,0.7))
        
    elif adf['Basket_Size'].get(i) == 'big':
        if adf['Day'].get(i) == 'Saturday' or adf['Day'].get(i) == 'Sunday':
            adf['item_weight'].iloc[i] = np.random.choice(choice,p=(0.7,0.3))
            
        else:
            adf['item_weight'].iloc[i] = np.random.choice(choice,p=(0.5,0.5))
            
adf['item_weight'].iloc[0] = np.random.choice(choice)

# label encode the 'item weight'
binary_encoder = preprocessing.LabelBinarizer()
norminal_encoder = preprocessing.LabelEncoder()
ordinal_encoder = OrdinalEncoder()

adf['encoded_weight'] = int(0)
#st.write(adf['encoded_weight'])
#adf['encoded_weight'] = binary_encoder.fit_transform(adf['encoded_weight'])    
#adf['encoded_weight'] = adf["encoded_weight"].astype(int)

for k in adf.index:
    if adf['item_weight'].get(k) == 'light':
        adf['encoded_weight'].iloc[k] = int(1)

adf['encoded_weight'].iloc[0] = int(1)

st.write(adf['encoded_weight'].dtypes)
st.write(adf)

st.subheader('Merge the result back to cleaned_df and encoded_df \n Latest cleaned_df:')
cleaned_df['item_weight'] = adf['item_weight']
encoded_df['item_weight'] = adf['encoded_weight']

st.write(cleaned_df.head())

st.text('Latest encoded_df')
st.write(encoded_df.head())

st.header('Feature Selection')

X = encoded_df.drop(['item_weight','Date_Time'], axis='columns')
y = encoded_df['item_weight']

st.subheader('Feature Selection with BORUTA and Random Forest \n Boruta')

# Boruta
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=10)
feat_selector = BorutaPy(rf, n_estimators="auto", random_state=10)

feat_selector.fit(X.values, y.values.ravel())
colnames = X.columns

boruta_score = ranking(list(map(float, feat_selector.ranking_)), X.columns, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])

boruta_score = boruta_score.sort_values("Score", ascending = False)

########################################################################
top10_boruta=['Day','Age_Range','Shirt_Colour','Pants_Colour','Basket_colour','Age_Bin','Part_of_Day','Washer_No','Kids_Category','Dryer_No']
top10_score = [1.00,0.95,0.89,0.84,0.79,0.74,0.68,0.68,0.58,0.53]
bottom10_boruta = ['Race','Body_Size','Attire','Spectacles','Gender','pants_type','Wash_item','With_Kids','shirt_type','Basket_Size']
bottom10_score = [0.47,0.42,0.37,0.32,0.26,0.21,0.16,0.11,0.05,0.00]
index1 = [18,3,9,11,7,17,19,14,5,15]
index2 = [0,2,8,16,1,12,13,4,10,6]

top10 = {'':index1,'Feature': top10_boruta, 'Score':top10_score}

top10 = pd.DataFrame(top10)
top10.set_index('',inplace = True)

bottom10 = {'':index2, 'Feature': bottom10_boruta, 'Score':bottom10_score}

bottom10 = pd.DataFrame(bottom10)
bottom10.set_index('',inplace = True)

overall = {'':index1+index2,'Feature':top10_boruta + bottom10_boruta, 'Score': top10_score + bottom10_score}
overall = pd.DataFrame(overall)
overall.set_index('', inplace=True)
########################################################################

st.info('---------Boruta Top 10----------')
#st.write(boruta_score.head(10))
st.write(top10)

st.info('---------Boruta Bottom 10----------')
#st.write(boruta_score.tail(10))
st.write(bottom10)

st.info("Boruta Plot")
boruta_plot = sns.catplot(x="Score", y="Feature", data = overall, kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
plt.title("RFE Top "+str(20)+" Features")
st.pyplot()

st.text('Random Forest Elimination (RFE)')

rf2 = RandomForestClassifier(n_jobs=-1 , class_weight='balanced',max_depth=5, random_state=5)

rfe = RFECV(rf2, min_features_to_select=1,cv=5)
rfe.fit(X,y)
rfe_score = ranking(list(map(float, rfe.ranking_)), X.columns, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

######################################################################
rfetop10 = ['Day','Age_Range','Shirt_Colour','Pants_Colour','Basket_colour','Part_of_Day','Dryer_No','Age_Bin','Washer_No','Kids_Category',]
rfescoret10 = [1.00,1.00,1.00,0.94,0.88,0.82,0.76,0.71,0.65,0.59]

rfebottom10 = ['Race','Body_Size','Attire','Spectacles','Gender','With_Kids','shirt_type','pants_type','Wash_item','Basket_Size']
rfescoreb10 = [0.53,0.47,0.41,0.35,0.29,0.24,0.18,0.12,0.06,0.00]

index3 = [18,3,9,11,7,19,15,17,14,5]
index4 = [0,2,8,16,1,4,10,12,13,6]

rfe_top10 = {'':index3, 'Feature':rfetop10, 'Score':rfescoret10}
rfe_top10 = pd.DataFrame(rfe_top10)
rfe_top10.set_index('',inplace = True)

rfe_bottom10 = {'':index4, 'Feature':rfebottom10, 'Score':rfescoreb10}
rfe_bottom10 = pd.DataFrame(rfe_bottom10)
rfe_bottom10.set_index('',inplace = True)

overall_rfe = {'':index3+index4, 'Feature':rfetop10+rfebottom10, 'Score':rfescoret10+rfescoreb10}
overall_rfe = pd.DataFrame(overall_rfe)
overall_rfe.set_index('',inplace=True)

######################################################################


st.info('---------RFE Top 10----------')
#st.write(rfe_score.head(10))
st.write(rfe_top10)

st.info('---------RFE Bottom 10----------')
#st.write(rfe_score.tail(10))
st.write(rfe_bottom10)

st.info("RFE Plot")
sns_rfe_plot = sns.catplot(x="Score", y="Feature", data = overall_rfe, kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
plt.title("RFE Top "+str(20)+" Features")
st.pyplot()

boruta_feat = set(boruta_score[:15]['Features'])
rfe_feat = set(rfe_score[:15]['Features'])

boruta_rfe = boruta_feat.intersection(rfe_feat)

best_feat = list(boruta_rfe)
st.write('Number of boruta feature       : ',len(boruta_feat))
st.write('Number of RFE feature          : ',len(rfe_feat))
st.write('Number of Boruta & RFE feature : ',15)

st.write()
st.write('Number of best feature         : ',15)

best_feat = ['Basket_colour', 'Gender', 'Age_Range', 'Part_of_day', 'Body_Size', 'Shirt_Colour', 'Attire', 'Spectacles', 'Age_Bin', 'Washer_No', 'Kids_Category', 'Day', 'Pants_Colour', 'Race', 'Dryer_No']

st.write(best_feat)

st.subheader('Apply Train-Test Split on dataset with top best features')
## DROP COLS that has similar feature from feature selection 
st.write('Initial to train features: ', len(X.columns))

st.write('\nBest features: ',len(best_feat))
st.write('List of best features :\n',best_feat)
to_drop = np.setdiff1d(list(X.columns),best_feat)

st.write('\nTo drop features: ',len(to_drop))
st.write('List of to drop features :\n',to_drop)
X = X.drop(to_drop, axis='columns')

st.subheader('Train-Test Split (Without SMOTE)')
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=5)
st.write('\nTo train features: ', len(X.columns))
st.write('List of To train features :\n',X.columns)

st.write("\nThe number of rows in the train set (Without SMOTE)  : ", X_train.shape)
st.write("The number of rows in the  test set (Without SMOTE)  : ", X_test.shape)

show_unique = [ [] for i in range(X_train.shape[1])]
for i in range(X_train.shape[1]):
    show_unique[i].append(X.columns[i])
    show_unique[i].append(len(np.unique(X_train.iloc[:,i])))
    show_unique[i].append(list(np.unique(X_train.iloc[:,i])))
show_unique_df = pd.DataFrame(show_unique,columns=['Attribute','Count','Data'])
show_unique_df.set_index('Attribute',inplace=True)
st.write(show_unique_df)

st.subheader('Train-Test Split (With SMOTE)')

smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=5, k_neighbors=5)
X_res, y_res = smt.fit_resample(X, y)
X_trainS, X_testS, y_trainS, y_testS = train_test_split(X_res, y_res, test_size = 0.30, random_state = 10)

st.write("The number of rows in the train set (With SMOTE)     : ", X_trainS.shape)
st.write("The number of rows in the  test set (With SMOTE)     : ", X_testS.shape)

SMOTE_unique = [ [] for i in range(X_trainS.shape[1])]
for i in range(X_trainS.shape[1]):
    SMOTE_unique[i].append(X.columns[i])
    SMOTE_unique[i].append(len(np.unique(X_trainS.iloc[:,i])))
    SMOTE_unique[i].append(list(np.unique(X_trainS.iloc[:,i])))
SMOTE_unique_df = pd.DataFrame(SMOTE_unique,columns=['Attribute','Count','Data'])
SMOTE_unique_df.set_index('Attribute',inplace=True)
st.write(SMOTE_unique_df)

st.subheader('Compare dataset shape counter')
st.write('Original  dataset shape %s' % Counter(y))
st.write('Resampled dataset shape %s' % Counter(y_res))

st.header('Machine Learning')
st.subheader('K-Mode Clustering \n - assign some cluster to customer')

X_temp = encoded_df.copy()
X_temp = X_temp[["Race","shirt_type","Attire","Basket_Size","Wash_Item"]]
st.write(X_temp.head())

st.subheader('Cao')
cost = []
for num_clusters in list(range(1,10)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao",verbose=0,random_state=10)
    kmode.fit_predict(X_temp)
    cost.append(kmode.cost_)
label = np.arange(1,10)
plt.plot(label,cost)
st.pyplot()
st.text("based on the default kmode use, graph shows there's 3 clusters base on elbow method")

km_cao = KModes(n_clusters=3, verbose=0,random_state=10)
fitClusters_cao = km_cao.fit_predict(X_temp,y)

clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = X_temp.columns

st.write('Silhouette coefficient :' , silhouette_score(X_temp,fitClusters_cao))
sv = SilhouetteVisualizer(km_cao)
sv.fit(X_temp)
st.write(sv.poof())
st.pyplot()


st.subheader('Huang')
cost = []
for num_clusters in list(range(1,10)):
    kmode = KModes(n_clusters=num_clusters, init = "Huang",verbose=0,random_state=10)
    kmode.fit_predict(X_temp)
    cost.append(kmode.cost_)
label = np.arange(1,10)
plt.plot(label,cost)
st.pyplot()

km_huang = KModes(n_clusters=3, init = 'Huang',verbose=1,random_state=10)
fitClusters_huang = km_huang.fit_predict(X_temp)

st.write('Silhouette coefficient :' , silhouette_score(X_temp,fitClusters_huang))
sv = SilhouetteVisualizer(km_huang)
sv.fit(X_temp)
st.write(sv.poof())
st.pyplot()

st.text('K-Mode : Huang is better than the default cluster which is cao. Hence, Huang is use')

clustersDf = pd.DataFrame(fitClusters_huang)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([X.reset_index(), clustersDf], axis = 1)
combinedDf = combinedDf.set_index('index')
#combinedDf[combinedDf["cluster_predicted"].isnull()]  
combinedDf['Part_of_day'] = combinedDf['Part_of_day'].map(
                            {0 : 'Early Morning',
                            1 : 'Morning',
                            2 : 'Afternoon',
                            3 : 'Evening'})
combinedDf['Day'] = combinedDf['Day'].map(
                            {0 : 'Monday',
                            1 : 'Tuesday',
                            2 : 'Wednesday',
                            3 : 'Thursday',
                            4 : 'Friday',
                            5 : 'Saturday',
                            6 : 'Sunday'
                            })
st.write(combinedDf)

clustersDf.isnull().any()

combinedDf.cluster_predicted.value_counts()

plt.subplots(figsize = (15,5))
sns.countplot(x=combinedDf['Part_of_day'],order=combinedDf['Part_of_day'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()
st.pyplot()

plt.subplots(figsize = (15,5))
sns.countplot(x=combinedDf['Day'],order=combinedDf['Day'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()
st.pyplot()

st.header('Classification \n - KNN \n - RandomForest Classifier \n - Gaussian Naive Bayes Classifier \n - Decision Tree Classifier \n - Support Vector Classifier')

# Import libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Model evaluation 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

#append value for overall model evaluation
fpr   = []
tpr   = []
auc   = []
pre   = []
rec   = []
label = []

fprS   = []
tprS   = []
aucS   = []
preS   = []
recS   = []
labelS = []

# function
def visual_heatmap(y_test, y_pred,model_name='Model_name'):
    cf = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                  cf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                      cf.flatten()/np.sum(cf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    accuracy  = np.trace(cf) / float(np.sum(cf))
    precision = cf[1,1] / sum(cf[:,1])
    recall    = cf[1,1] / sum(cf[1,:])
    f1_score  = 2*precision*recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(accuracy,precision,recall,f1_score)

    sns.heatmap(cf, annot=labels,fmt ='')
    plt.title(model_name + ' Model Evaluation')
    plt.xlabel('Predicted label'+ stats_text)
    plt.ylabel('Ground truth label')
    st.pyplot()

def visual_TP_FP(y_test, proba,model_name='model_name'):
    plt.figure(figsize=(5,5))
    fpr, tpr, thresholds = roc_curve(y_test, proba) 
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr,tpr,label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve for ' + model_name)
    plt.show()
    st.pyplot()
    return fpr,tpr,auc

def visual_precision_recall(y_test, proba,model_name='model_name'):
    plt.figure(figsize=(5,5))
    precision, recall, thresholds = precision_recall_curve(y_test, proba) 
    plt.plot(precision, recall,label = model_name)
    plt.legend(loc = 'lower left')
    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for ' + model_name)
    plt.show()
    st.pyplot()
    return precision, recall

def evaluate_model(model,model_name,X_train, y_train,X_test, y_test):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    proba = proba[:,1]

    # print("Accuracy on training set: {:.2f} %".format(model.score(X_train, y_train)*100))
    # print("Accuracy on test set    : {:.2f} %".format(model.score(X_test, y_test)*100))
    # print('********************************************')
    # print('Accuracy   = {:.2f} %'. format(accuracy_score(y_test, y_pred)*100))
    # print('Precision  = {:.2f} %'. format(precision_score(y_test, y_pred,average='micro')*100))
    # print('Recall     = {:.2f} %'. format(recall_score(y_test, y_pred,average='micro')*100))
    # print('F1         = {:.2f} %'. format(f1_score(y_test, y_pred,average='micro')*100))

    visual_heatmap(y_test, y_pred,model_name)
    fpr,tpr,auc = visual_TP_FP(y_test, proba,model_name)
    pre,rec = visual_precision_recall(y_test, proba,model_name)
    return fpr,tpr,auc,pre,rec

st.subheader('KNN')
st.subheader('Without SMOTE')
k_range = range(1,20)
scores = []

for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    accuracy = knn.score(X_test, y_test)
    scores.append(accuracy)

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Accuracy by n_neigbors Without SMOTE')
plt.scatter(k_range, scores)
plt.xticks(np.arange(20))
plt.plot(k_range, scores, color='green', linestyle='dashed', linewidth=1, markersize=5)
st.pyplot()

st.info('KNN Without SMOTE')
knn_classifiers = KNeighborsClassifier(n_neighbors=10)
knn_classifiers.fit(X_train,y_train)

fpr_knn,tpr_knn,auc_knn,pre_knn,rec_knn = evaluate_model(knn_classifiers,'KNN',X_train, y_train,X_test, y_test)

st.subheader('With SMOTE')
st.info('KNN WitH SMOTE')
k_range = range(1,20)
scores = []

for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainS,y_trainS)
    accuracy = knn.score(X_testS, y_testS)
    scores.append(accuracy)

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Accuracy by n_neigbors With SMOTE')
plt.scatter(k_range, scores)
plt.xticks(np.arange(20))
plt.plot(k_range, scores, color='green', linestyle='dashed', linewidth=1, markersize=5)
st.pyplot()

st.info('KNN With SMOTE')
knn_classifiers = KNeighborsClassifier(n_neighbors=2)
knn_classifiers.fit(X_trainS,y_trainS)
fpr_knnS,tpr_knnS,auc_knnS,pre_knnS,rec_knnS = evaluate_model(knn_classifiers,'KNN SMOTE',X_trainS, y_trainS,X_testS, y_testS)

fpr.append(fpr_knn)
tpr.append(tpr_knn)
auc.append(auc_knn)
pre.append(pre_knn)
rec.append(rec_knn)
label.append('KNN')

fprS.append(fpr_knnS)
tprS.append(tpr_knnS)
aucS.append(auc_knnS)
preS.append(pre_knnS)
recS.append(rec_knnS)
labelS.append('KNN w SMOTE')

st.subheader('Random Forest Classifier')
st.subheader("Without SMOTE")

from sklearn.metrics import mean_absolute_error

rf_range = range(1,20)
maes = []

for d in range(1,20):
    rf = RandomForestClassifier(max_depth = d, random_state = 10)
    rf.fit(X_train, y_train)
    y_rf_predict = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_rf_predict)
    maes.append(rf_mae)

plt.figure()
plt.xlabel('d')
plt.ylabel('MAE')
plt.title('MAE on Random Forest Classifier Without SMOTE')
plt.scatter(rf_range, maes)
plt.xticks(np.arange(20))
plt.plot(rf_range, maes, color='green', linestyle='dashed', linewidth=1, markersize=5)
st.pyplot()

st.info('Random Forest Classifier Without SMOTE')
rf = RandomForestClassifier(max_depth=2, random_state = 10)
rf.fit(X_train, y_train)
fpr_rf,tpr_rf,auc_rf,pre_rf,rec_rf = evaluate_model(rf,'Random Forest without SMOTE',X_train, y_train,X_test, y_test)

st.subheader('With SMOTE')
rf_range = range(1,20)
maes = []

for d in range(1,20):
    rf = RandomForestClassifier(max_depth = d, random_state = 10)
    rf.fit(X_trainS, y_trainS)
    y_rf_predict = rf.predict(X_testS)
    rf_mae = mean_absolute_error(y_testS, y_rf_predict)
    maes.append(rf_mae)

plt.figure()
plt.xlabel('d')
plt.ylabel('MAE')
plt.title('MAE on Random Forest Classifier With SMOTE')
plt.scatter(rf_range, maes)
plt.xticks(np.arange(20))
plt.plot(rf_range, maes, color='green', linestyle='dashed', linewidth=1, markersize=5)
st.pyplot()

st.info('Random Forest Classifier With SMOTE')
rf = RandomForestClassifier(max_depth=18, random_state = 10)
rf.fit(X_trainS, y_trainS)
fpr_rfS,tpr_rfS,auc_rfS,pre_rfS,rec_rfS = evaluate_model(rf,'Random Forest with SMOTE',X_trainS, y_trainS,X_testS, y_testS)

fpr.append(fpr_rf)
tpr.append(tpr_rf)
auc.append(auc_rf)
pre.append(pre_rf)
rec.append(rec_rf)
label.append('Random Forest')

fprS.append(fpr_rfS)
tprS.append(tpr_rfS)
aucS.append(auc_rfS)
preS.append(pre_rfS)
recS.append(rec_rfS)
labelS.append('Random Forest w SMOTE')

st.subheader('Decision Tree Classifier')
st.subheader('Without SMOTE')

from sklearn.model_selection import GridSearchCV

param_grid = {
              'max_depth' : np.arange(5,25,1)
              }

grid = GridSearchCV(DecisionTreeClassifier(criterion='gini',random_state=10,max_features ='auto'),param_grid,cv=5)
grid.fit(X_train,y_train)
st.write('The best parameters are %s with 5-CV score of %0.2f' 
        % (grid.best_params_,grid.best_score_))

st.info('Gini Tree without SMOTE')
decision_tree_gini = DecisionTreeClassifier(criterion='gini',
                                       max_features ='auto',
                                       max_depth=7,
                                       random_state=10)
decision_tree_gini.fit(X_train,y_train)
fpr_gini,tpr_gini,auc_gini,pre_gini,rec_gini = evaluate_model(decision_tree_gini,'Gini Tree',X_train, y_train,X_test, y_test)    


st.info('Gini Tree with SMOTE')
decision_tree_gini = DecisionTreeClassifier(criterion='gini',
                                       max_features ='auto',
                                       max_depth=15,
                                       random_state=10)
decision_tree_gini.fit(X_trainS,y_trainS)
fpr_giniS,tpr_giniS,auc_giniS,pre_giniS,rec_giniS = evaluate_model(decision_tree_gini,'Gini Tree',X_trainS, y_trainS,X_testS, y_testS)


fpr.append(fpr_gini)
tpr.append(tpr_gini)
auc.append(auc_gini)
pre.append(pre_gini)
rec.append(rec_gini)
label.append('Gini Tree')

fprS.append(fpr_giniS)
tprS.append(tpr_giniS)
aucS.append(auc_giniS)
preS.append(pre_giniS)
recS.append(rec_giniS)
labelS.append('Gini Tree w SMOTE')

st.subheader('Support Vector Classifier')
st.subheader('Without SMOTE')
param_grid = {'kernel' : ['rbf'],
              'C' : [0.001, 0.01, 0.1, 1, 10],
              'gamma' : [0.001, 0.01, 0.1, 1]
              }


grid = GridSearchCV(SVC(random_state=10),param_grid,cv=5)
grid.fit(X_train,y_train)
st.write('The best parameters are %s with 5-CV score of %0.2f' 
        % (grid.best_params_,grid.best_score_))

st.info('SVC Without SMOTE')
svm_classifier = SVC(kernel = 'rbf', C=10 , gamma =0.001, probability=True , random_state=10)
svm_classifier.fit(X_train,y_train)

fpr_svc,tpr_svc,auc_svc,pre_svc,rec_svc = evaluate_model(svm_classifier,'SVC',X_train, y_train,X_test, y_test)


st.subheader('With SMOTE')
param_grid = {'kernel' : ['rbf'],
              'C' : [0.001, 0.01, 0.1, 1, 10],
              'gamma' : [0.001, 0.01, 0.1, 1]
              }


grid = GridSearchCV(SVC(random_state=10),param_grid,cv=5)
grid.fit(X_trainS,y_trainS)
st.write('The best parameters are %s with 5-CV score of %0.2f' 
        % (grid.best_params_,grid.best_score_))

st.info('SVC With SMOTE')
svm_classifier = SVC(kernel = 'rbf',C = 10, gamma = 0.01, probability=True, random_state=10)
svm_classifier.fit(X_trainS,y_trainS)

fpr_svcS,tpr_svcS,auc_svcS,pre_svcS,rec_svcS = evaluate_model(svm_classifier,'SVC',X_trainS, y_trainS,X_testS, y_testS)


fpr.append(fpr_svc)
tpr.append(tpr_svc)
auc.append(auc_svc)
pre.append(pre_svc)
rec.append(rec_svc)
label.append('SVC')

fprS.append(fpr_svcS)
tprS.append(tpr_svcS)
aucS.append(auc_svcS)
preS.append(pre_svcS)
recS.append(rec_svcS)
labelS.append('SVC w SMOTE')

st.subheader('Naive Bayes')
st.subheader('Without SMOTE')

param_grid = {'var_smoothing': np.arange(1e-15,1e-5,1e-1)
              }

grid = GridSearchCV(GaussianNB(),param_grid,cv=5)
grid.fit(X_train,y_train)
st.write('The best parameters are %s with 5-CV score of %0.2f' 
        % (grid.best_params_,grid.best_score_))

st.info('Naive Bayes without SMOTE')
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
fpr_nb,tpr_nb,auc_nb,pre_nb,rec_nb = evaluate_model(naive_bayes,'Naive Bayes',X_train, y_train,X_test, y_test)


st.subheader('With SMOTE')
param_grid = {'var_smoothing': np.arange(1e-15,1e-5,1e-1)
              }

grid = GridSearchCV(GaussianNB(),param_grid,cv=5)
grid.fit(X_trainS,y_trainS)
st.write('The best parameters are %s with 5-CV score of %0.2f' 
        % (grid.best_params_,grid.best_score_))

st.info('Naive Bayes with SMOTE')
naive_bayes = GaussianNB()
naive_bayes.fit(X_trainS, y_trainS)
fpr_nbS,tpr_nbS,auc_nbS,pre_nbS,rec_nbS = evaluate_model(naive_bayes,'Naive Bayes w SMOTE',X_trainS, y_trainS,X_testS, y_testS)


fpr.append(fpr_nb)
tpr.append(tpr_nb)
auc.append(auc_nb)
pre.append(pre_nb)
rec.append(rec_nb)
label.append('Naive Bayes')

fprS.append(fpr_nbS)
tprS.append(tpr_nbS)
aucS.append(auc_nbS)
preS.append(pre_nbS)
recS.append(rec_nbS)
labelS.append('Naive Bayes w SMOTE')

st.subheader('Overall Classifier Performance')
st.info('Overall analysis without SMOTE')
plt.figure(figsize=(5,5))
for i in range(len(fpr)):
  st.write(label[i] + '\t AUC : ' + str(auc[i]))
  plt.plot(fpr[i], tpr[i], label=label[i] + ' AUC: %.2f '%(auc[i]*100) + '%') 

plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve w/o SMOTE')
plt.legend()
st.pyplot()

plt.figure(figsize=(5,5))
for i in range(len(pre)):
    plt.plot(pre[i], rec[i],label = label[i])
plt.legend(loc = 'lower left')
plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve w/o SMOTE')
st.pyplot()

st.info('Overall analysis with SMOTE')
plt.figure(figsize=(5,5))
for i in range(len(fprS)):
    print(labelS[i] + '\t AUC : ' + str(aucS[i]))
    plt.plot(fprS[i], tprS[i], label=labelS[i] + ' AUC: %.2f '%(aucS[i]*100) + '%') 

plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve w SMOTE')
plt.legend()
st.pyplot()

plt.figure(figsize=(5,5))
for i in range(len(preS)):
    plt.plot(preS[i], recS[i],label = labelS[i])
plt.legend(loc = 'lower left')
plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve w SMOTE')
st.pyplot()

st.header('Save the best model! \n - Random Forest with SMOTE')

import pickle

st.info('Random Forest Classifier With SMOTE')
rf_final = RandomForestClassifier(max_depth=18, random_state = 10)
rf_final.fit(X_trainS, y_trainS)
fpr_rfS,tpr_rfS,auc_rfS,pre_rfS,rec_rfS = evaluate_model(rf_final,'Random Forest with SMOTE',X_trainS, y_trainS,X_testS, y_testS)

pkl_filename = "random_forest_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(rf_final, file)

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

score = pickle_model.score(X_test, y_test)
st.write("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(X_test)
