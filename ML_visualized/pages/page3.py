import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

def app():
    st.write("Data visualization")

    # Function to load data from a CSV file based on the selected option
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
            try:
                data = pd.read_csv(file_path)
                return data
            except FileNotFoundError:
                st.error(f"File '{file_path}' not found.")
                return None
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None
        else:
            st.error("Invalid option selected")
            return None

    # Page 3 content
    st.title("Page 3: Data Visualization")

    # Ensure that selected_type and selected_data are set
    if 'selected_type' not in st.session_state or st.session_state['selected_type'] is None:
        st.write("Please go back to Page 1 and select an option.")
        return
    elif 'selected_data' not in st.session_state or st.session_state['selected_data'] is None:
        st.write("Please go back to Page 2 and select an option.")
        return

    selected_option = st.session_state['selected_data']
    
    # Load data based on the selected option
    data = load_data(selected_option)
    
    if data is not None:
        st.write(f"Data loaded for {selected_option}")
        
        # Display data table
        st.write(data.head())

        # Display data description
        st.write("## Data Description")
        st.write(data.describe())

        # Display data info
        st.write("## Data Information")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Visualization functions
        st.write("## Visualizations")

        # Example of a bar chart
        st.write("### Bar Chart of the First Column")
        plt.figure(figsize=(10, 5))
        data.iloc[:, 0].value_counts().plot(kind='bar')
        plt.xlabel('Categories')
        plt.ylabel('Counts')
        plt.title('Bar Chart of the First Column')
        st.pyplot(plt)

        # Example of a heatmap
        st.write("### Heatmap of Correlations")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=.5)
        plt.title('Heatmap of Correlations')
        st.pyplot(plt)

    
        # Additional visualizations can be added here

    else:
        st.error("Failed to load data.")

