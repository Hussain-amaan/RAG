import streamlit as st

#from rag_utils import process_file


st.set_page_config(page_title="ADVANCE RAG",layout="wide")
st.title("ADVANCE RAG")

st.sidebar.header("upload file")

uploaded_file=st.sidebar._file_uploader(
    "choose file",
    type=["pdf","txt","csv"],
    accept_multiple_files=True)

chunk_size=st.sidebar.number_input(
    "select chunk size",
    min_value=1000,
    max_value=5000,
    value=1000 , 
    step= 100)


chunk_overlap=st.sidebar.number_input(
    "chunk overlap",
    min_value=0,
    max_value=500,
    value=100 , 
    step= 10)


top_k=st.sidebar.number_input(
    "Document to retrieve  per query",
    min_value=1,
    max_value=10,
    value=3 )  


if st.sidebar.button("submit"):
    if uploaded_file:
        st.spinner("processing...")
        process_file(uploaded_file,chunk_size,chunk_overlap)
    else:
        st.warning("please upload file first ")

