import streamlit as st

def app():
    st.write("First, let's start with selection of learning type")

    # Initialize session state to store selected type
    if 'selected_type' not in st.session_state:
        st.session_state['selected_type'] = None

    # Function to handle button click
    def select_data(data):
        st.session_state['selected_type'] = data
        st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h4>Regression</h4>
            <p>Regression task is used to create a model based on continuous data</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Regression"):
            select_data("Regression")

    with col2:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h4>Classification</h4>
            <p>Classification task is used to create a model based on discrete data</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Classification"):
            select_data("Classification")

    # Move to Page 2 if data is selected
    if st.session_state['selected_type']:
        st.write("Data selected:", st.session_state['selected_type'])
        st.write("Navigate to select data page to select ML data.")
