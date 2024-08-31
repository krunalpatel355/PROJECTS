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

    selected_type = 'Regression'

    st.write(f"Selected type: {selected_type}")

    # Model options based on selected type
    if selected_type == "Regression":
        options = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor", "K-Nearest Neighbors Regressor"]
    elif selected_type == "Classification":
        options = ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier", "K-Nearest Neighbors Classifier"]
    
    # Model selection
    options = 'Linear Regression'
    selected_model = st.selectbox("Choose a model", options)
    
    # Number of loops input
    num_loops = st.number_input("Number of loops", min_value=1, value=1, step=1)
    
    # Data split inputs
    st.write("Data Split")
    col1, col2 = st.columns(2)
    with col1:
        train_split = st.number_input("Training Data (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    with col2:
        test_split = st.number_input("Testing Data (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    
    # Ensure that train_split + test_split == 100%
    if train_split + test_split != 100.0:
        st.error("Training and testing splits must add up to 100%")
        return

    if st.button("Submit"):
        st.write("Loading data...")

        # Load data from the selected CSV file
        selected_data = 'house'
        data = load_data(selected_data)
        if data is None:
            return

        # Split data into features and target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_split/100), random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Run the model execution function
        model_exe(selected_model, X_train, X_test, y_train, y_test, num_loops)
        st.write("Loading data... done!")

def load_data(selected_option):
    # Define a mapping from options to CSV file paths
    option_to_file = {
        "House": "datasets/house.csv",
        "Diamonds": "datasets/diamonds.csv",
        "Flower": "datasets/flower.csv",
        "Cancer": "datasets/cancer.csv",
    }

        # Get the file path based on the selected option
    file_path = option_to_file.get(selected_option)

    if file_path:
        with open(file_path, 'r') as file:
            data = pd.read_csv(file)
        return data
    else:
        st.error("Invalid option selected")
        return None

def model_exe(model_name, X_train, X_test, y_train, y_test, epochs):
    if 'Regression' in model_name:
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Decision Tree Regressor":
            model = DecisionTreeRegressor(max_depth=2, random_state=42)
        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "Support Vector Regressor":
            model = SVR(kernel='linear')
        elif model_name == "K-Nearest Neighbors Regressor":
            model = KNeighborsRegressor(n_neighbors=5)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.write(f'Model Performance:')
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Mean Absolute Error: {mae}')

    elif 'Classification' in model_name:
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Decision Tree Classifier":
            model = DecisionTreeClassifier(max_depth=2, random_state=42)
        elif model_name == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "Support Vector Classifier":
            model = SVC(kernel='linear')
        elif model_name == "K-Nearest Neighbors Classifier":
            model = KNeighborsClassifier(n_neighbors=5)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(y_pred)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f'Model Performance:')
        st.write(f'Accuracy: {accuracy}')
