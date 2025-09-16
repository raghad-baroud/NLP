"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd


st.markdown("# Expo Visitor's Review Analysis")
st.markdown("This website provides an automated sentiment classification and analysis of Expo's visitors feedback")
st.text("");st.text("")
st.markdown("#### Upload File Here")

uploaded_file=st.file_uploader('Supported formats: CSV, XLSX, TXT',type=['csv','xlsx','txt'])

def is_valid(data):
  data=data.iloc[:,-1]

  if data.ndim!=1 or not isinstance(data[0],str):
    st.error('Please Enter dataset of texts only.')
    return None
  return data


if uploaded_file is not None:
  file_format=uploaded_file.name.split(".")[-1]
  if file_format=='csv' or file_format=='txt':
    d=is_valid(pd.read_csv(uploaded_file))
    if(not d is None):
      st.session_state.df=d
  elif file_format=='xlsx':
    d=is_valid(pd.read_excel(uploaded_file))
    if(not d is None):
      st.session_state.df=d
  else:
    st.markdown("***The uploaded file is not supported***")


