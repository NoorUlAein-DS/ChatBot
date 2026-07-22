import streamlit as st
import joblib
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# 1. Page Configuration
st.set_page_config(page_title="Heart Health Predictor & AI Doctor", layout="wide")
st.title("🫀 Heart Health Predictor & AI Medical Assistant")

# 2. Load ML Model (Decision Tree / KNN)
@st.cache_resource
def load_ml_model():
    try:
        # Update file name if using KNN (e.g., 'knn_model.pkl') or DT ('tree_model.pkl')
        return joblib.load("heart_model.pkl")
    except Exception as e:
        return None

model = load_ml_model()

# 3. Create 2-Column Layout
col1, col2 = st.columns([1, 1])

# --- LEFT COLUMN: HEART DISEASE PREDICTION ---
with col1:
    st.header("1. Heart Disease Risk Prediction")
    st.write("Enter the patient's medical details below:")
    
    # Input Fields (Simple English Labels)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
    bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    
    if st.button("Predict Risk"):
        if model is not None:
            user_data = np.array([[age, bp, chol]])
            prediction = model.predict(user_data)[0]
            
            if prediction == 1:
                st.error("⚠️ **High Risk Detected!** Please consult a cardiologist immediately.")
            else:
                st.success("✅ **Low Risk Detected!** Patient parameters appear stable.")
        else:
            st.warning("⚠️ Model file ('heart_model.pkl') not found in the repository!")

# --- RIGHT COLUMN: LANGCHAIN AI DOCTOR CHATBOT ---
with col2:
    st.header("2. AI Medical Assistant")
    st.write("Ask any health questions or request definitions for medical terms:")

    # Input Box in Simple English
    user_query = st.text_input("Enter your query (e.g., What is Cholesterol?):")

    if st.button("Ask Doctor AI"):
        if not user_query:
            st.warning("Please type a question first!")
        else:
            with st.spinner("Doctor thinking..."):
                try:
                    # Fetches API Key automatically from Streamlit Secrets
                    llm = ChatGroq(
                        model='llama-3.1-8b-instant',
                        temperature=0.7,
                        api_key=st.secrets["GROQ_API_KEY"]
                    )
                    
                    # System Prompt enforcing English responses
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful and empathetic medical doctor. Answer health questions in very simple, easy-to-understand English with clear examples."),
                        ("human", "{user_question}")
                    ])
                    
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({"user_question": user_query})
                    
                    st.info(response)
                except Exception as e:
                    st.error(f"Error: {e}")
