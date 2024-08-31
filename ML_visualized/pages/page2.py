import streamlit as st

def app():
    st.write("Second, let's select the dataset for prediction")

    # Ensure that selected_type is set
    if 'selected_type' not in st.session_state or st.session_state['selected_type'] is None:
        st.write("Please go back to Page 1 and select an option.")
        return

    selected_data = st.session_state['selected_type']

    st.write(f"Selected type: {selected_data}")

    # Define options based on selected data type
    if selected_data == "Regression":
        options = ["House", "Diamonds"]
    elif selected_data == "Classification":
        options = ["Flower", "Cancer"]

    selected_option = st.selectbox("Choose an option", options)
    if st.button("Select data"):
        st.session_state['selected_data'] = selected_option
        
    st.write(f"You selected: {st.session_state.get('selected_data', 'None')}")
