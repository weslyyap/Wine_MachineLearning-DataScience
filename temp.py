# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

st.write("Wine Customer Segment Classifier")

#Define sub functions
def Data_Fetching(file_path, num):
    
    if num == 1:
        data = pd.read_csv(file_path)
        data = data.dropna()
        st.write(data)
        st.write("Rows, Columns: ", data.shape)
        return data
        
    elif num == 2:
        data1 = pd.read_csv(file_path)
        data1 = data1.dropna()
        st.write(data1)
        st.write("Rows, Columns: ", data1.shape)
        return data1

def Data_Cleaning(data, data1, num):
    
    if num == 1:
        white_wine = data.loc[(data["type"] == "white")]
        red_wine = data.loc[(data["type"] == "red")]
        row_count_white = white_wine.shape[0]
        row_count_red = red_wine.shape[0]
        
        #Take the smaller data sets
        if row_count_white >= row_count_red:
            rows = row_count_red
        else:
            rows = row_count_white
        
        #Randomly taking data sets
        white_wine = white_wine.sample(rows)
        red_wine = red_wine.sample(rows)
        
        #Join two separates df back to data
        data = pd.concat([white_wine, red_wine])
        return data
    
    elif num == 2:
        #Get the distribution of customer segment
        print(data1['Wine'].unique())
        cus1 = data1[data1['Wine'] == 1]
        cus1 = cus1.sort_values(by = 'Alcohol', ascending = True)
        cus1 = cus1.reset_index(drop = True)
        cus2 = data1[data1['Wine'] == 2]
        cus2 = cus2.sort_values(by = 'Alcohol', ascending = True)
        cus2 = cus2.reset_index(drop = True)
        cus3 = data1[data1['Wine'] == 3]
        cus3 = cus3.sort_values(by = 'Alcohol', ascending = True)
        cus3 = cus3.reset_index(drop = True)
        
        #Restrict the size of data as data1
        data1_size = data1.shape[0]
        data = data.sample(data1_size)
        
        #Reset the index of data
        data = data.reset_index(drop = True)
        
        #Cannot directly join because each customer segment isn't based on the alcohol%
        #Segment into each customer segment in data for data1
        data_cs1 = data.iloc[0:cus1.shape[0]]
        data_cs1 = data_cs1.sort_values(by = 'alcohol', ascending = True)
        data_cs1 = data_cs1.reset_index(drop = True)
        a = cus1.shape[0]
        data_cs2 = data.iloc[a : (a + cus2.shape[0])]
        data_cs2 = data_cs2.sort_values(by = 'alcohol', ascending = True)
        data_cs2 = data_cs2.reset_index(drop = True)
        b = a + cus2.shape[0]
        data_cs3 = data.iloc[b : (b + cus3.shape[0])]
        data_cs3 = data_cs3.sort_values(by = 'alcohol', ascending = True)
        data_cs3 = data_cs3.reset_index(drop = True)
        
        #Merging both Datasets
        merged_cs1 = pd.concat([cus1, data_cs1], axis = 1)
        merged_cs2 = pd.concat([cus2, data_cs2], axis = 1)
        merged_cs3 = pd.concat([cus3, data_cs3], axis = 1)
        newData = pd.concat([merged_cs1, merged_cs2, merged_cs3])
        newData = newData.reset_index(drop = True)
        
        #Replace both alcohol values with mean value for new dataset
        newData['Alcohol%'] = (newData['Alcohol'] + newData['alcohol'])/2
        newData = newData.drop(columns = ['Alcohol', 'alcohol'])
        
        #One hot encoding
        newData = pd.get_dummies(newData, drop_first = True)
        cols = list(newData)
        cols.insert(0, cols.pop(cols.index('type_white'))) # 1 = white wine, 0 = red wine
        newData = newData.loc[:, cols]
        newData.rename({'type_white' : 'wine type'}, axis = 'columns', inplace = True)
        newData.rename({'Wine' : 'Customer'}, axis = 'columns', inplace = True)
        output = newData.pop('Customer')
        newData.insert(newData.shape[1], 'Customer', output)
        return newData
    

def Data_Visualization(flag):
    
    plt.clf()
    if flag:
        bc = data.groupby(["type"]).mean().plot.bar(stacked=True, cmap="RdYlBu", figsize=(15, 5))
        bc.set_xlabel("Wine Type", fontsize = 18)
        bc.set_ylabel("Chemical Distributions", fontsize = 18)
        plt.rcParams["font.family"] = "sans-serif"
        plt.suptitle(
        "Average Chemical Distribution Across Red And White Wine", fontsize=20
        )
        plt.xticks(size=18, rotation="horizontal")
        #plt.xlabel("Wine Type", fontsize=18)
        #plt.ylabel("Chemical Distributions", fontsize=18)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.48), ncol=4, fontsize=15)
        st.pyplot(bc.figure)
 
    plt.clf()
    plt.rcParams["font.family"] = "sans-serif"
    plt.suptitle("Distribution of Wine Quality and Color", fontsize=20)
    plt.xlabel("xlabel", fontsize=18)
    plt.ylabel("ylabel", fontsize=16)
    ax = sns.countplot(
       x="quality", hue="type", data=data, palette=["#f9dbbd", "#da627d"])
    ax.set(xlabel="Wine Quality Rating", ylabel="Number of Bottles")
    st.pyplot(ax.figure)      
    return

def Data_Preparation():
    st.write("Normalizing Data...")
    X = newData.drop(['Customer'], axis = 1)
    Y = newData['Customer']
    X_features = X
    X = StandardScaler().fit_transform(X)
    
    st.write("Splitting Data...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=0)
    return X_train, X_test, Y_train, Y_test


def Data_Modelling(X_train, X_test, Y_train, Y_test):
    #Using Linear Regression Model  
    Lmodel = linear_model.LogisticRegression(random_state = 0)
    Lmodel.fit(X_train, Y_train)
    Y_predL = Lmodel.predict(X_test)
    st.write("Preicting results...")
    
    #Data Visualization On Linear Regression Model's Performance
    #Seaborn plot
    plt.clf()
    sp = sns.scatterplot(x = Y_test, y = Y_predL, hue = Y_predL, palette = ['orange', 'dodgerblue', 'green'], size = Y_predL, legend = True)
    sp.set(title = 'Distributions of Actual Value vs Predicted Value (Logistic Regression)', xlabel = 'Customer Segment', ylabel = 'Predicted Customer Segment')
    sp.legend(title = 'Customer Segment')
    st.pyplot(sp.figure)
    
    #Mathplotlib plot
    # construct cmap
    flatui = ["#9b59b6", "#3498db", "#95a5a6"]
    my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    N = 3
    #data1 = np.random.randn(N)
    #data2 = np.random.randn(N)
    colors = {'red', 'green', 'yellow'}
    scatter = plt.scatter(Y_test, Y_predL, c=Y_predL, cmap=my_cmap)
    #plt.legend(labels = ["1", "2", "3"], ncol = 3, bbox_to_anchor = (2 , 1), title = 'Customer Segment')
    plt.legend(*scatter.legend_elements(), title = 'Customer Segment', bbox_to_anchor = (1.32 , 1))
    plt.colorbar()
    plt.show()
    st.pyplot(scatter.figure)
    
    #Confusion matrix
    st.write("Confusion Matrix...")
    cm = confusion_matrix(Y_test, Y_predL)
    cm
    st.write("Model's accuracy(%): ", (accuracy_score(Y_test,Y_predL) * 100))
    return Y_predL

    
#Start executing
st.write("Fetching 1st Dataset from Database...")
#file_path = "C:\\Users\\Weihau.yap\\Desktop/winequalityN.csv"
file_path = 'winequalityN.csv'
data = Data_Fetching(file_path, 1)

st.write("Fetching 2nd Dataset from Database...")
#file_path = "C:\\Users\\Weihau.yap\\Desktop/wine.csv"
file_path = 'wine.csv'
data1 = Data_Fetching(file_path, 2)

st.write("Checking Data Distributions...")
Data_Visualization(True)

st.write("Data Cleaning to ensure well distributed wine type...")
data = Data_Cleaning(data, data1, 1)
st.write(data)
st.write("Rows, Columns: ", data.shape)

st.write("Evenly Distributed Wine type...")
Data_Visualization(False)

st.write("Cleaning and Merging Both Datasets...")
newData = Data_Cleaning(data, data1, 2)
st.write(newData)
st.write("Rows, Columns: ", newData.shape)

st.write("Preparing Dataset for builidng ML model...")
X_train, X_test, Y_train, Y_test = Data_Preparation()

st.write("Training built ML model...")   
Y_predL = Data_Modelling(X_train, X_test, Y_train, Y_test)
  
# Then graph the distribution so we see how may red vs white bottles we have
plt.clf()
plt.rcParams["font.family"] = "sans-serif"
plt.suptitle("Distribution of Customers and Wine Type", fontsize=20)
plt.xlabel("xlabel", fontsize=18)
plt.ylabel("ylabel", fontsize=16)

ax = sns.countplot(
    x="Customer", hue="wine type", data=newData, palette=["#3498db", "#95a5a6"]
)
ax.set(xlabel="Customer Segment", ylabel="Number of Customers")
ax.legend(title = "Wine Type")
st.pyplot(ax.figure)


    