from pandasai.llm.local_llm import LocalLLM ## Importing LocalLLM for local Meta Llama 3 model
import streamlit as st 
import pandas as pd # Pandas for data manipulation
from pandasai import SmartDataframe # SmartDataframe for interacting with data using LLM


model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="llama3"
)

st.title("Data analysis with Pandas AI and Llama 3 Model")

input_files = st.file_uploader("Upload your CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True)

if input_files:
    # Select a file from the uploaded files using a dropdown menu
    selected_file = st.selectbox("Select a file", [file.name for file in input_files])
    selected_index = [file.name for file in input_files].index(selected_file)

    # Load and display the selected file
    st.info("File uploaded successfully")
    file_extension = selected_file.split('.')[-1]

    st.text("Head (3)")

    try:
        if file_extension == 'csv':
            data = pd.read_csv(input_files[selected_index])
        elif file_extension == 'xlsx':
            data = pd.read_excel(input_files[selected_index])

        st.dataframe(data.head(3), use_container_width=True)
    
        df = SmartDataframe(data,{"enable_cache": False},config={"llm": model})
        
        prompt = st.text_area("What do you want to ask?")

        if st.button("Ask"):
            if prompt:
                with st.spinner("Generating Request..."):
                    st.write(df.chat(prompt))

    except Exception as e:
        st.error(f"Error: {e}")

#make sure you already installed llama3 using ollama
#To use type in powershell "cd \path", then type "streamlit run app2.py"
