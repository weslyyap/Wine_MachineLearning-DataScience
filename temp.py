# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
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
import requests


#Define sub functions
def Data_Fetching(file_path, num):
    
    if num == 1:
        data = pd.read_csv(file_path)
        data = data.dropna()
        #st.write(data)
        #st.write("Rows, Columns: ", data.shape)
        return data
        
    elif num == 2:
        data1 = pd.read_csv(file_path)
        data1 = data1.dropna()
        #st.write(data1)
        #st.write("Rows, Columns: ", data1.shape)
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
    st.subheader("Predicting results...")
    
    #Data Visualization On Linear Regression Model's Performance
    #Seaborn plot
    plt.clf()
    sp = sns.scatterplot(x = Y_test, y = Y_predL, hue = Y_predL, palette = ['orange', 'dodgerblue', 'green'], size = Y_predL, legend = True)
    sp.set(title = 'Distributions of Actual Value vs Predicted Value (Logistic Regression)', xlabel = 'Customer Segment', ylabel = 'Predicted Customer Segment')
    sp.legend(title = 'Customer Segment')
    st.pyplot(sp.figure)
    
    #Mathplotlib plot
    # construct cmap
    # flatui = ["#9b59b6", "#3498db", "#95a5a6"]
    # my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    # N = 3
    # #data1 = np.random.randn(N)
    # #data2 = np.random.randn(N)
    # colors = {'red', 'green', 'yellow'}
    # scatter = plt.scatter(Y_test, Y_predL, c=Y_predL, cmap=my_cmap)
    # #plt.legend(labels = ["1", "2", "3"], ncol = 3, bbox_to_anchor = (2 , 1), title = 'Customer Segment')
    # plt.legend(*scatter.legend_elements(), title = 'Customer Segment', bbox_to_anchor = (1.32 , 1))
    # plt.colorbar()
    # plt.show()
    # st.pyplot(scatter.figure)
    
    #Confusion matrix
    st.subheader("Confusion Matrix...")
    cm = confusion_matrix(Y_test, Y_predL)
    cm
    st.write("Model's accuracy(%): ", (accuracy_score(Y_test,Y_predL) * 100))
    return Y_predL

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    
#Set up front end
#Webpage config
st.set_page_config(page_title = "Wine With Me!", page_icon=":wine_glass:", layout = "wide")
st.title("Wine Customer Segment Classifier ML Model")
st.write("---")

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Abstract", "Data Visualization", "Model Prediction"],
        icons = ["eyeglasses", "bar-chart-line", "hourglass-split"],
        default_index = 0,
        )
    
#Start executing
#file_path = "C:\\Users\\Weihau.yap\\Desktop/winequalityN.csv"
file_path = 'winequalityN.csv'
data = Data_Fetching(file_path, 1)

#file_path1 = "C:\\Users\\Weihau.yap\\Desktop/wine.csv"
file_path1 = 'wine.csv'
data1 = Data_Fetching(file_path1, 2)


if selected == "Abstract":
    #Header section
    with st.container():
        
        left_column, center_column, right_column  = st.columns(3)
        with left_column:
            lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_7psw7qge.json")
            st_lottie(lottie_hello, key = "hello", speed = 1.3, width = 200)

        with center_column:
            #lottie_wine = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_ztpmvmp2.json")
            lottie_wine = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_6eroa6if.json")
            st_lottie(lottie_wine, key = "wine", speed = 0.6, width = 200)
            
        with right_column:
            st.subheader("Highlights of this project:")
            st.write("""
                     1. Create new datasets from multiple sources (Kaggle).
                     2. Classify wine into different customer segments.
                     3. Understand our target customers!
                     """)
                     
    #st.write("This project is to build a Wine Recommender System")
    st.subheader("Fetching 1st Dataset from Database...")
    st.write("[Kaggle Source >] https://www.kaggle.com/datasets/rajyellow46/wine-quality")
    st.write(data)
    st.write("Rows, Columns: ", data.shape)

    st.subheader("Fetching 2nd Dataset from Database...")
    st.write("[Kaggle Source >] https://gist.github.com/tijptjik/9408623")
    st.write(data1)
    st.write("Rows, Columns: ", data1.shape)

elif selected == "Data Visualization":
    st.subheader("Checking Data Distributions...")
    Data_Visualization(True)

    st.subheader("Data Cleaning to ensure well distributed wine type...")
    data = Data_Cleaning(data, data1, 1)
    st.write(data)
    st.write("Rows, Columns: ", data.shape)

    st.subheader("Evenly Distributed Wine type...")
    Data_Visualization(False)

    st.subheader("Cleaning and Merging Both Datasets...")
    newData = Data_Cleaning(data, data1, 2)
    st.write(newData)
    st.write("Rows, Columns: ", newData.shape)

elif selected == "Model Prediction":
    newData = Data_Cleaning(data, data1, 2)
    st.subheader("Preparing Dataset for builidng ML model...")
    X_train, X_test, Y_train, Y_test = Data_Preparation()

    st.write("Training built ML model...")
    with st.container():
        
        left_column, right_column = st.columns(2)
        with left_column:
            lottie_ml = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_m075yjya.json")
            st_lottie(lottie_ml, key = "ml", width = 300)
            lottie_arrow = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_dbwrpcsu.json")
            st_lottie(lottie_arrow, key = "arrow", width = 100)
            
        with right_column:
            Y_predL = Data_Modelling(X_train, X_test, Y_train, Y_test)

    #Header section
    st.subheader("So, who's our target customer? :thought_balloon:")
    st.write("####")
    with st.container():
        
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("Seems like white wine is more favorable in the market.")
            st.write("The percentage of white wine is higher across all the customers!")
            #lottie_good = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_ktzgqvov.json")
            lottie_good1 = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_wdplkjoi.json")
            #st_lottie(lottie_good, key = "good", width = 300)
            st_lottie(lottie_good1, key = "good1", width = 300)

        with right_column:
            # Then graph the distribution so we see how may red vs white bottles we have
            plt.clf()
            plt.rcParams["font.family"] = "sans-serif"
            plt.suptitle("Distribution of Customers and Wine Type", fontsize=12)
            plt.xlabel("xlabel", fontsize=10)
            plt.ylabel("ylabel", fontsize=10)

            ax = sns.countplot(
                x="Customer", hue="wine type", data=newData, palette=["#3498db", "#95a5a6"]
            )
            ax.set(xlabel="Customer Segment", ylabel="Number of Customers")
            ax.legend(title = "Wine Type")
            st.pyplot(ax.figure)
    

    



    
    


    