import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

def app():
    st.write("Select model type and configure training parameters")

    # Ensure that selected_type and selected_data are set
    if 'selected_type' not in st.session_state or st.session_state['selected_type'] is None:
        st.write("Please go back to Page 1 and select an option.")
        return
    if 'selected_data' not in st.session_state or st.session_state['selected_data'] is None:
        st.write("Please go back to Page 2 and select an option.")
        return
    
    selected_type = st.session_state['selected_type']
    selected_data = st.session_state['selected_data']
    
    st.write(f"Selected type: {selected_type}")

    # Model options based on selected type
    if selected_type == "Regression":
        options = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor", "K-Nearest Neighbors Regressor"]
    elif selected_type == "Classification":
        options = ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier", "K-Nearest Neighbors Classifier"]
    
    # Model selection
    selected_model = st.selectbox("Choose a model", options)
    st.write(f"Selected model: {selected_model}")

    # Number of loops input
    num_loops = st.number_input("Number of loops", min_value=1, value=1, step=1)
    
    # Data split inputs
    st.write("Data Split")
    col1, col2 = st.columns(2)
    with col1:
        train_split = st.number_input("Training Data (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    with col2:
        test_split = st.number_input("Testing Data (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    
    # Ensure that train_split + test_split == 100%
    if train_split + test_split != 100.0:
        st.error("Training and testing splits must add up to 100%")
        return

    if st.button("Submit"):
        st.write("Loading data...")

    path = 'datasets/' + selected_data + '.csv'
    df = pd.read_csv(path)

    # st.write(df.head())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_split)/100, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    if selected_type == 'Regression':
        if selected_model == "Linear Regression":
            model = LinearRegression()
        elif selected_model == "Decision Tree Regressor":
            model = DecisionTreeRegressor()
        elif selected_model == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=1000,max_depth = 3, random_state=42)
        elif selected_model == "Support Vector Regressor":
            model = SVR()
        elif selected_model == "K-Nearest Neighbors Regressor":
            model = KNeighborsRegressor(n_neighbors=10)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

                # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.write(f'Model Performance:')
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Mean Absolute Error: {mae}')

        
    else:
        if selected_model == "Logistic Regression":
            model = LogisticRegression()
        elif selected_model == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif selected_model == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=1000,max_depth = 3, random_state=42)
        elif selected_model == "Support Vector Classifier":
            model = SVC()
        elif selected_model == "K-Nearest Neighbors Classifier":
            model = KNeighborsClassifier(n_neighbors=10)
        else:
            raise ValueError("Unsupported model selected")
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f'Model Performance:')
        st.write(f'Accuracy: {accuracy}')




