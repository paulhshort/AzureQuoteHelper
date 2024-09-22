import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import plotly.express as px

# Load environment variables
load_dotenv()

# Configuration
AZURE_PRICING_API_URL = "https://prices.azure.com/api/retail/prices"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1" if OPENROUTER_API_KEY else "https://api.openai.com/v1",
    api_key=OPENROUTER_API_KEY or OPENAI_API_KEY,
)

@st.cache_data
def extract_text_from_file(file):
    try:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError("Unsupported file format")
        return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_data
def get_azure_pricing(service_name):
    try:
        response = requests.get(f"{AZURE_PRICING_API_URL}?$filter=serviceName eq '{service_name}'")
        return response.json()
    except Exception as e:
        st.error(f"Error querying Azure Pricing API: {e}")
        return None

def query_ai_model(prompt, model="openai/gpt-4o-2024-08-06"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in Azure VM pricing and recommendations."},
                {"role": "user", "content": prompt}
            ],
            extra_headers={
                "HTTP-Referer": "https://your-app-url.com",
                "X-Title": "AzurePricingApp",
            } if OPENROUTER_API_KEY else {},
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying AI model: {e}")
        return None

def analyze_quote_and_recommend(file_text):
    prompt = (
        f"Given the following quote, analyze the virtual machine configurations and performance requirements. "
        f"Then recommend the most suitable Azure VM sizes and SKUs, considering cost-efficiency and performance. "
        f"Provide detailed reasons for your recommendations. Format your response as a JSON object with the following structure: "
        f"{{'recommendations': [{{'vm_size': 'string', 'sku': 'string', 'reasons': ['string'], 'estimated_cost': 'string'}}]}}.\n\nQuote:\n{file_text}"
    )
    recommendations = query_ai_model(prompt)
    return json.loads(recommendations) if recommendations else None

def visualize_recommendations(recommendations):
    df = pd.DataFrame(recommendations['recommendations'])
    
    # Bar chart for estimated costs
    fig_cost = px.bar(df, x='vm_size', y='estimated_cost', title='Estimated Costs by VM Size')
    st.plotly_chart(fig_cost)
    
    # Radar chart for VM characteristics
    # (You would need to extract or generate more data points for a meaningful radar chart)
    # This is just a placeholder example
    fig_radar = px.line_polar(df, r=[1, 2, 3, 4], theta=['CPU', 'Memory', 'Storage', 'Network'], line_close=True)
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar)

def main():
    st.set_page_config(page_title="Azure Pricing and VM Recommendations", layout="wide")
    st.title("Azure Pricing and VM Recommendations")

    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    
    if uploaded_file:
        file_text = extract_text_from_file(uploaded_file)
        
        if file_text:
            with st.spinner("Analyzing quote and generating recommendations..."):
                recommendations = analyze_quote_and_recommend(file_text)
                if recommendations:
                    st.subheader("Recommendations")
                    for rec in recommendations['recommendations']:
                        with st.expander(f"VM Size: {rec['vm_size']} - SKU: {rec['sku']}"):
                            st.write(f"**Estimated Cost:** {rec['estimated_cost']}")
                            st.write("**Reasons:**")
                            for reason in rec['reasons']:
                                st.write(f"- {reason}")
                    
                    visualize_recommendations(recommendations)
    
    st.sidebar.header("Azure Pricing Lookup")
    service_name = st.sidebar.text_input("Enter Azure Service Name")
    if st.sidebar.button("Fetch Pricing"):
        with st.spinner("Fetching pricing data..."):
            pricing_data = get_azure_pricing(service_name)
            if pricing_data:
                df = pd.DataFrame(pricing_data['Items'])
                st.sidebar.dataframe(df[['skuName', 'unitPrice', 'productName']])

    # Chat interface
    st.header("Chat with Azure Pricing Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about Azure pricing or VM recommendations"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = query_ai_model(prompt)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()