# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRFRegressor
import joblib
import streamlit as st
import datetime as dt


# Step 1: Data Preprocessing Function
def preprocess_data(data):
    # Handle missing values for 'Item_Weight'
    data['Item_Weight_interploate'] = data['Item_Weight'].interpolate(method="linear")
    data = data.drop(['Item_Weight'], axis=1)

    # Impute 'Outlet_Size' based on 'Outlet_Type'
    mode_outlet = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
    missing_values = data['Outlet_Size'].isnull()
    data.loc[missing_values, 'Outlet_Size'] = data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_outlet[x])

    # Replace inconsistencies in 'Item_Fat_Content'
    data.replace({'Item_Fat_Content': {'Low Fat': 'LF', 'low fat': 'LF', 'reg': 'Regular'}}, inplace=True)

    # Interpolate 'Item_Visibility'
    data['Item_Visibility_interpolate'] = data['Item_Visibility'].replace(0, np.nan).interpolate(method='linear')
    data = data.drop('Item_Visibility', axis=1)

    # Encode 'Item_Identifier'
    data['Item_Identifier'] = data['Item_Identifier'].apply(lambda x: x[:2])

    # Calculate 'Outlet_age'
    current_year = dt.datetime.today().year
    data['Outlet_age'] = current_year - data['Outlet_Establishment_Year']
    data = data.drop('Outlet_Establishment_Year', axis=1)

    # Encode categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        oe = OrdinalEncoder()
        data[col] = oe.fit_transform(data[[col]])

    return data


# Step 2: Model Training Function
def train_model(data):
    X = data.drop('Item_Outlet_Sales', axis=1)
    y = data['Item_Outlet_Sales']
    final_data = X.drop(columns=['Item_Visibility_interpolate', 'Item_Weight_interploate',
                                 'Item_Type', 'Outlet_Location_Type', 'Item_Identifier', 'Item_Fat_Content'], axis=1)

    xg = XGBRFRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(xg, final_data, y, cv=5, scoring='r2')
    print(f"Cross-validated R2 Score: {scores.mean()}")

    # Train the model on the full dataset
    xg_final = XGBRFRegressor(n_estimators=100, random_state=42)
    xg_final.fit(final_data, y)
    # xg_final = XGBRFRegressor()
    # xg_final.fit(final_data, y)
    #
    # X_train, X_test, y_train, y_test = train_test_split(final_data, y,
    #                                                     test_size=0.20,
    #                                                     random_state=42)
    # xg_final.fit(X_train, y_train)
    # xg = XGBRFRegressor(n_estimators=100, random_state=42)
    # scores = cross_val_score(xg, final_data, y, cv=5, scoring='r2')
    # print(f"Cross-validated R2 Score: {scores.mean()}")
    #
    # # # Train the model on the full dataset
    # # xg_final = XGBRFRegressor(n_estimators=100, random_state=42)
    # # xg_final.fit(X, y)

    return xg_final


# Step 3: Save the Model Function
def save_model(model, filename='bigmart_model.pkl'):
    joblib.dump(model, filename)


# Step 4: Streamlit App Function
def build_streamlit_app():
    model = joblib.load('bigmart_model.pkl')

    # Get the current year
    current_year = dt.datetime.today().year

    # Streamlit app title
    st.title("Big Mart Sales Prediction using Machine Learning")

    # User inputs
    p1 = st.number_input("Enter Item MRP:", min_value=0.0, format="%.2f")

    outlet_identifier_options = ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019',
                                 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049']
    text = st.selectbox("Select Outlet Identifier:", outlet_identifier_options)

    outlet_size_options = ['High', 'Medium', 'Small']
    text0 = st.selectbox("Select Outlet Size:", outlet_size_options)

    outlet_type_options = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']
    text1 = st.selectbox("Select Outlet Type:", outlet_type_options)

    outlet_establishment_year = st.number_input("Enter Outlet Establishment Year:", min_value=1900,
                                                max_value=current_year)

    # Convert inputs to numeric values
    p2 = outlet_identifier_options.index(text)
    p3 = outlet_size_options.index(text0)
    p4 = outlet_type_options.index(text1)
    p5 = current_year - int(outlet_establishment_year)

    # Predict button
    if st.button("Predict"):
        result = model.predict(np.array([[p1, p2, p3, p4, p5]]))[0]

        lower_bound = float(result) - 714.42
        upper_bound = float(result) + 714.42

        st.write(f"Sales Amount is in between {lower_bound:.2f} and {upper_bound:.2f}")


# Main function to run the entire pipeline
def main():
    # Load the data
    data = pd.read_csv('big_mart_Train.csv')

    # Step 1: Preprocess the data
    data_preprocessed = preprocess_data(data)

    # Step 2: Train the model
    model = train_model(data_preprocessed)

    # Step 3: Save the model
    save_model(model)

    # Step 4: Build and run the Streamlit app
    build_streamlit_app()


if __name__ == "__main__":
    main()


