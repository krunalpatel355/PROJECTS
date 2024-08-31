import streamlit as st
import pages.page1 as page1
import pages.page2 as page2
import pages.page3 as page3
import pages.page4 as page4

# Title of the app
st.title("MACHINE LEARNING VISUALIZE")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", ("Select learning type", "Select data", "Data visualization", "Select model type"))

# Navigation logic
if options == "Select learning type":
    page1.app()
elif options == "Select data":
    page2.app()
elif options == "Data visualization":
    page3.app()
elif options == "Select model type":
    page4.app()
