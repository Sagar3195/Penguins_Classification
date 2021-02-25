import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
## Penguins Classification App

This app clssify the **Penguins Species**
""")

st.sidebar.header("User Input Features")
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

#Collect user input features into dataframe

uploaded_file = st.sidebar.file_uploader("Upload input csv file:", type = ['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_feature():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox("Sex", ("Male", 'Female'))
        bill_length_mm = st.sidebar.slider("Bill Length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider("Body Mass (g)", 2700.0, 6300.0, 4207.0)

        data = {"island": island,
                "bill_length_mm": bill_length_mm,
                "bill_depth_mm": bill_depth_mm,
                "flipper_length_mm": flipper_length_mm,
                "body_mass_g": body_mass_g,
                "sex": sex        
        }

        feature = pd.DataFrame(data, index = [0])
        return feature
    input_df = user_input_feature()

#Combined user input features with entire penguins dataset
#This will be useful for encoding phase
penguins_df = pd.read_csv("penguins_cleaned.csv")
#Here we dropping species columns as target variable
penguins = penguins_df.drop(columns = ['species'])
#penguins 
concat_df = pd.concat([input_df, penguins], axis = 0)
#concat_df 
#Now encoding categorical features
concat_df = pd.get_dummies(concat_df)
#concat_df
#Select only first rows(the user input data)
new_df = concat_df[:1]
#new_df
 
##Displays the use input features
st.subheader("User Input Features")

if uploaded_file is not None:
    st.write(new_df)
else:
    st.write("Awaiting CSV file to be uploaded.Currently using example input parameters (show below).")
    st.write(new_df)

##Load the pickle
clf = pickle.load(open("penguins_clf_model.pkl", 'rb'))

##Now make prediction
prediction = clf.predict(new_df)
prediction_prob = clf.predict_proba(new_df)

st.subheader("Prediction")
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader("Prediction Probability")
st.write(prediction_prob)