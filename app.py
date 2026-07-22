import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Page Configuration
st.set_page_config(page_title="Heart Health Predictor & AI Doctor", layout="wide")
st.title("🫀 Heart Health Predictor & AI Medical Assistant")

# Layout Split
col1, col2 = st.columns([1, 1])

# --- LEFT COLUMN: ML MODEL PREDICTION ---
with col1:
    st.header("1. Heart Disease Prediction")
    st.write("Apni medical details enter karein:")
    
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    
    if st.button("Predict Disease Risk"):
        # Yahan aapka model predict karega
        st.success("Result: Low Risk (Example Output)")

# --- RIGHT COLUMN: AI DOCTOR CHATBOT ---
# RIGHT COLUMN: AI DOCTOR CHATBOT
with col2:
    st.header("2. AI Medical Assistant")
    st.write("Medical terms samajhne ke liye AI Doctor se poochein:")

    user_query = st.text_input("Aapka Sawal:")

    if st.button("Ask Doctor AI"):
        if not user_query:
            st.warning("Please type a question!")
        else:
            with st.spinner("Doctor thinking..."):
                try:
                    # Streamlit Secrets se API Key le rahe hain
                    llm = ChatGroq(
                        model='llama-3.1-8b-instant',
                        temperature=0.7,
                        api_key=st.secrets["GROQ_API_KE"]
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful medical doctor. Answer health questions in very simple, easy language with clear examples."),
                        ("human", "{user_question}")
                    ])
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({"user_question": user_query})
                    st.info(response)
                except Exception as e:
                    st.error(f"Error: {e}")
