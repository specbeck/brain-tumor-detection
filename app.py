import streamlit as st
import json
from google import genai

# ==========================================
# SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Neuro-oncology AI Assistant", page_icon="🧠", layout="centered")

# Drop your Gemini API key here
GOOGLE_API_KEY = "AQ.Ab8RN6IidIG29OU7m64U9MRi6UizSUki4f5fxPDCa2y_gxr8-w" 

@st.cache_data
def load_context():
    try:
        with open('./assistant/context.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "context.json not found. Please ensure it exists in the /assistant directory."}

project_context = load_context()

# ==========================================
# UI LAYOUT
# ==========================================
st.title("🧠 ML Architecture Co-Pilot")
st.markdown("""
Welcome to the interactive engineering documentation for the **Brain Tumor Classification** project. 
Ask me to defend the architectural choices, explain the 163-dimensional feature space, or break down the clinical safety metrics.
""")

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., Why did you use an SVM instead of a CNN?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Construct the context-aware prompt
    system_prompt = f"""
    You are the lead Machine Learning Engineer who designed this neuro-oncology diagnostic pipeline. 
    Defend your engineering choices using this exact context:
    
    {json.dumps(project_context, indent=2)}
    
    Guidelines for your response:
    1. Be highly technical, confident, and concise. 
    2. If asked about a topic not directly in the context but related to ML or the pipeline (e.g., standard ML practices, general metrics, biological definitions of the tumors), extrapolate logically as a senior engineer would, but always tie it back to the project's constraints (CPU-only, clinical safety).
    3. Do not break character. 
    
    User question: {prompt}
    """
    
    # Call the API
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt
        )
        reply = response.text
    except Exception as e:
        reply = f"**API Error:** Ensure your Gemini API Key is valid. Details: `{e}`"
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(reply)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": reply})
