# ==============================================================
# Author: Adrian Rodriguez
# 
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script has been originally created by Adrian Rodriguez.
# Any explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ==============================================================

# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# Intro
st.title("Star Classifier Report")
st.write(
    """
    Welcome throughout this repot I will summarize the proces of
    my Star Classifier proyect.
    """
)

# Business understanding
st.header("Business understanding")
st.markdown(
    """
    The SDSS has been searching for DataScientist and they ask you to join their team.

    For this job you need to gather data from de SDSS survey and create a star classifier system.

    In this project I will follow the **CRISP-DM model**

    The CRoss Industry Standard Process for Data Mining (CRISP-DM) is a process model that serves as the base for a data science process. It has six sequential phases:

    - Business understanding – What does the business need?
    - Data understanding – What data do we have / need? Is it clean?
    - Data preparation – How do we organize the data for modeling?
    - Modeling – What modeling techniques should we apply?
    - Evaluation – Which model best meets the business objectives?
    - Deployment – How do stakeholders access the results?
    
    """
    )

st.header("Data understanding ")
st.markdown(
    """
    ## Collect initial data
    I couldn't create a connection between the ipynb and the SDSS 
    RestApi to retrevie the data becasue is not available any more, 
    what I can do to gather the data is to consult the Cassjob 
    online data base and download it in csv.
    """
)

st.image('img/Query.jpeg', caption='Query from Cassjojob')

# Load data
dr14 = pd.read_csv('input/Skyserver_adradev_DR14.csv')
st.dataframe(dr14.sample(5))

st.write(
    """
    We can observe that we have a class variable in the data set, 
    this will be usefull as our objective variable.
    Let's see what it can tell us to build our model
    """
)

df2 = pd.DataFrame(dr14)
df2.drop(['rerun', 'objid'], axis=1, inplace=True)
df2 = df2.rename(columns={'class':'Class'})
Class = df2.Class.astype('category')

# Plotting
st.subheader("Ploting the different objects, Galaxy, Star, QSO")
st.vega_lite_chart(
    df2,
    {
        "mark": {"type": "bar", "tooltip":True},
        "encoding":{
            "x": {"field": "Class", "Type":"nominal"},
            "y": {"agregate": "count", "field": "field", "type":"quantitative"},
        "color":{"field": "Class", "type":"nominal"},
        },
    },
)

st.write(
    """
    We can conclude that we have quality data, and it exist a strong 
    relationship between the dec(decendance) and u, g, r, i, z 
    variables.
    """
)

# Data preparation
st.header("Data preparation")
st.subheader("Select data")
st.markdown(
    """ 
    To select the data, I will perform a PCA to determine which
    columns are unnecessary.
    >**Note:** you cand find out the full proces in the notebook:
    (https://github.com/adra-dev/star_clasifier)
    """
)

st.subheader("Format data")
st.markdown(
    """
    ```python
    def drop_columns(df):
        dc=['rerun', 'objid', 'ra', 'fiberid','field','plate']
        for col in sorted(df.columns):
            if col in dc:
                df.drop(col, axis=1, inplace=True)
        return df
    ```

    >**Note:** The purpose of this function is create new data set 
    replacing the outlier data using the clip function

    ```
    def tweak_dr(dr14):
        drop_columns(dr14)
        return(dr14
        .rename(columns={'class': 'Class'})
        .assign(dec= lambda df_: df_.dec.clip(lower= dr14.dec.quantile(.25),
                                            upper= dr14.dec.quantile(.75)),
                u= lambda df_: df_.u.clip(lower= dr14.u.quantile(.25),
                                        upper= dr14.u.quantile(.75)),
                g= lambda df_: df_.g.clip(lower= dr14.g.quantile(.25),
                                        upper= dr14.g.quantile(.75)),
                r= lambda df_: df_.r.clip(lower= dr14.r.quantile(.25),
                                        upper= dr14.r.quantile(.75)),
                i= lambda df_: df_.i.clip(lower= dr14.i.quantile(.25),
                                        upper= dr14.i.quantile(.75)),
                z= lambda df_: df_.z.clip(lower= dr14.z.quantile(.25),
                                        upper= dr14.z.quantile(.75)),
                camcol= lambda df_: df_.camcol.clip(lower= dr14.camcol.quantile(.25),
                                                    upper= dr14.camcol.quantile(.75),),
        )

  )
    """
)

def drop_columns(df):
        dc=['rerun', 'objid', 'ra', 'fiberid','field','plate']
        for col in sorted(df.columns):
            if col in dc:
                df.drop(col, axis=1, inplace=True)
        return df


def tweak_dr(dr14):
    drop_columns(dr14)
    return(dr14
        .rename(columns={'class': 'Class'})
        .assign(dec= lambda df_: df_.dec.clip(lower= dr14.dec.quantile(.25),
                                            upper= dr14.dec.quantile(.75)),
                u= lambda df_: df_.u.clip(lower= dr14.u.quantile(.25),
                                        upper= dr14.u.quantile(.75)),
                g= lambda df_: df_.g.clip(lower= dr14.g.quantile(.25),
                                        upper= dr14.g.quantile(.75)),
                r= lambda df_: df_.r.clip(lower= dr14.r.quantile(.25),
                                        upper= dr14.r.quantile(.75)),
                i= lambda df_: df_.i.clip(lower= dr14.i.quantile(.25),
                                        upper= dr14.i.quantile(.75)),
                z= lambda df_: df_.z.clip(lower= dr14.z.quantile(.25),
                                        upper= dr14.z.quantile(.75)),
                camcol= lambda df_: df_.camcol.clip(lower= dr14.camcol.quantile(.25),
                                                    upper= dr14.camcol.quantile(.75),),
        )
)

# Modeling
st.header("Modeling")
st.subheader("Select modeling technique")
st.markdown(
    """ 
    Supervised learning learns a function to make prediction of a 
    defined label based on the input data. It can be either 
    classifying data into a category (classification problem) or 
    forecasting an outcome (regression algorithms).

    For this reason I will use and compare the next five models:

        - 1 Desicion Tree
        - 2 K nearest neighbour
        - 3 Naive bayes
        - 4 Random Forest
        - 5 Support Vector Machine

    """
)

# Generate tests design
dr14=tweak_dr(dr14)
X = dr14.drop('Class', axis=1)
y = dr14.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1)

# Feature Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Build model

st.subheader("Desicion Tree")
dt = DecisionTreeClassifier(criterion="gini", max_depth=3)
dt = dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))


Evaluation=pd.DataFrame(['Decision Tree'],columns=['Algorithm'])
Evaluation.loc[0,'Precision']=metrics.precision_score(y_test, y_pred, average='micro')
Evaluation.loc[0,'Recall']=metrics.recall_score(y_test, y_pred, average='micro')
Evaluation.loc[0,'F1 Score']=metrics.f1_score(y_test, y_pred, average='micro')
Evaluation.loc[0,'Accuracy']=metrics.accuracy_score(y_test,y_pred)



st.subheader("KNN")
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
y_pred=knn.predict(X_test)
st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))


Evaluation.loc[1,'Algorithm']='KNN'
Evaluation.loc[1,'Precision']=metrics.precision_score(y_test, y_pred, average='micro')
Evaluation.loc[1,'Recall']=metrics.recall_score(y_test, y_pred, average='micro')
Evaluation.loc[1,'F1 Score']=metrics.f1_score(y_test, y_pred, average='micro')
Evaluation.loc[1,'Accuracy']=metrics.accuracy_score(y_test,y_pred)



st.subheader("Naive bayes")
nb = GaussianNB()
nb.fit(X_train, y_train)
nb.score(X_test,y_test)
y_pred=nb.predict(X_test)
st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))

Evaluation.loc[2,'Algorithm']='Naive Bayes'
Evaluation.loc[2,'Precision']=metrics.precision_score(y_test, y_pred, average='micro')
Evaluation.loc[2,'Recall']=metrics.recall_score(y_test, y_pred, average='micro')
Evaluation.loc[2,'F1 Score']=metrics.f1_score(y_test, y_pred, average='micro')
Evaluation.loc[2,'Accuracy']=metrics.accuracy_score(y_test,y_pred)



st.subheader("Random Forest")
classifier1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier1.fit(X_train, y_train)
y_pred=classifier1.predict(X_test)
st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))

Evaluation.loc[3,'Algorithm']='Random Forest'
Evaluation.loc[3,'Precision']=metrics.precision_score(y_test, y_pred, average='micro')
Evaluation.loc[3,'Recall']=metrics.recall_score(y_test, y_pred, average='micro')
Evaluation.loc[3,'F1 Score']=metrics.f1_score(y_test, y_pred, average='micro')
Evaluation.loc[3,'Accuracy']=metrics.accuracy_score(y_test,y_pred)



st.subheader("Suport Vector Machine")
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))

Evaluation.loc[4,'Algorithm']='SVM'
Evaluation.loc[4,'Precision']=metrics.precision_score(y_test, y_pred, average='micro')
Evaluation.loc[4,'Recall']=metrics.recall_score(y_test, y_pred, average='micro')
Evaluation.loc[4,'F1 Score']=metrics.f1_score(y_test, y_pred, average='micro')
Evaluation.loc[4,'Accuracy']=metrics.accuracy_score(y_test,y_pred)

# Evaluation
st.header("Evaluation")
st.write(
    """
    Do the models meet the business success criteria? 
    Which one(s) should we approve for the business?
    """
)
Ev=Evaluation.sort_values(by='Accuracy' ,ascending=False)
Ev

st.write(
    """
    *   The Desicion Tree and the Random Forest are the best models that meet the business succes criteria, follow up by the SVM model.

    *   Imbalanced datasets are those where there is a severe skew in the class     
    distribution, such as 1:100 or 1:1000 examples in the minority class to the majority class, like in this case.

    *   This bias in the training dataset can influence many machine learning algorithms, leading some to ignore the minority class entirely. This is a problem as it is typically the minority class on which predictions are most important.

    *   Even after cliping the data to balance the result It’ll be worth it to dive deeper into label enconding and data scale with the purpouse of searching the best result by comparing of the ML models although Oversample and Undersample results.

    *  I bealive the reason for the Decision Tree to be the best algorithm and I quote:  **"Is Able to handle both numerical and categorical data."**

    * It's necesary to make a deeper data treatment.
    """ 
)
