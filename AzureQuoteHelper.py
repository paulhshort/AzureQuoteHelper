# AzureQuoteHelper.py

from __future__ import annotations

import streamlit as st
import requests
import json
import os
import re
from dotenv import load_dotenv
import openai
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import plotly.express as px
from pydantic import BaseModel, ValidationError
import logging

# Load environment variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(
    filename='azure_quote_helper.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Configuration
AZURE_PRICING_API_URL = "https://prices.azure.com/api/retail/prices"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment variables.")
    st.stop()

# Initialize OpenAI client via OpenRouter
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Define Pydantic model for Azure Pricing Function
class GetAzurePricing(BaseModel):
    vm_size: str
    disk_type: str  # e.g., 'Premium SSD Managed Disks'
    disk_size: str  # e.g., '400GB'
    region: str

# Mapping for user-friendly region names to Azure's armRegionName
regions_mapping = {
    'US East': 'eastus',
    'US East 2': 'eastus2',
    'US West': 'westus',
    'US West 2': 'westus2',
    'Europe West': 'westeurope',
    'South Central US': 'southcentralus',
    'East US 2': 'eastus2',
    'West Europe': 'westeurope',
    'Australia East': 'australeast',  # Corrected typo
    'Japan East': 'japaneast',
    'Southeast Asia': 'southeastasia',
    # Add more regions as needed
}

# Define Disk Types Mapping with accurate productName values
disk_product_mapping = {
    'Standard HDD': 'Standard HDD Managed Disks',
    'Standard SSD': 'Standard SSD Managed Disks',
    'Premium SSD': 'Premium SSD Managed Disks'
    # Add more disk types if necessary
}

# Define the function that fetches available VM sizes for a given service and region
@st.cache_data(ttl=3600)
def fetch_available_vm_sizes(service_name: str, region: str):
    """
    Fetch all available VM sizes (armSkuName) for the given service and region.
    """
    try:
        filter_query = f"serviceName eq '{service_name}' and armRegionName eq '{region}' and priceType eq 'Consumption'"
        params = {
            "$filter": filter_query,
            "$top": 1000,
            "api-version": "2023-01-01-preview"  # Updated API version
        }
        response = requests.get(AZURE_PRICING_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get('Items', [])
        sku_set = set()
        for item in items:
            if item.get('isPrimaryMeterRegion', False):
                sku = item.get('armSkuName', '')
                if sku:
                    sku_set.add(sku)
        sku_list = sorted(list(sku_set))
        logger.info(f"Fetched VM sizes for service '{service_name}' in region '{region}': {sku_list}")
        return sku_list
    except Exception as e:
        logger.error(f"Error fetching VM sizes: {e}")
        return []

# Define the function that fetches Azure pricing based on parameters
@st.cache_data(ttl=1800)  # Cache results for 30 minutes
def get_azure_pricing(vm_size: str, disk_type: str, disk_size: str, region: str):
    """
    Fetch pricing information for a specific VM size with disk specifications in a given region.

    Parameters:
    - vm_size (str): The size of the virtual machine (e.g., 'Standard_D4s_v3').
    - disk_type (str): The type of disk (e.g., 'Premium SSD Managed Disks').
    - disk_size (str): The size of the disk (e.g., '400GB').
    - region (str): The Azure region code (e.g., 'eastus').

    Returns:
    - dict: A dictionary containing VM price, Disk price, and Total estimated cost.
    """
    try:
        logger.info(f"Fetching pricing for VM Size: {vm_size}, Disk Type: {disk_type}, Disk Size: {disk_size}, Region: {region}")

        # Fetch VM pricing
        vm_filter = f"serviceName eq 'Virtual Machines' and armSkuName eq '{vm_size}' and armRegionName eq '{region}' and priceType eq 'Consumption'"
        vm_params = {
            "$filter": vm_filter,
            "$top": 1000,
            "api-version": "2023-01-01-preview"
        }
        vm_response = requests.get(AZURE_PRICING_API_URL, params=vm_params)
        vm_response.raise_for_status()
        vm_data = vm_response.json().get('Items', [])
        if not vm_data:
            error_msg = f"No VM pricing data found for size '{vm_size}' in region '{region}'."
            logger.error(error_msg)
            logger.debug(f"API Response: {vm_response.text}")
            return {"error": error_msg}

        # Assume the first matched item is the desired one
        vm_price = float(vm_data[0].get('unitPrice', 0))
        currency = vm_data[0].get('currencyCode', 'USD')
        logger.info(f"VM Price: {vm_price} {currency} per hour")

        # Fetch Disk pricing
        disk_product = disk_product_mapping.get(disk_type, disk_type)
        disk_filter = f"serviceName eq 'Storage' and productName eq '{disk_product}' and armRegionName eq '{region}' and priceType eq 'Consumption'"
        disk_params = {
            "$filter": disk_filter,
            "$top": 1000,
            "api-version": "2023-01-01-preview"
        }
        disk_response = requests.get(AZURE_PRICING_API_URL, params=disk_params)
        disk_response.raise_for_status()
        disk_data = disk_response.json().get('Items', [])
        if not disk_data:
            error_msg = f"No Disk pricing data found for type '{disk_type}' in region '{region}'."
            logger.error(error_msg)
            logger.debug(f"API Response: {disk_response.text}")
            return {"error": error_msg}

        # Assume the first matched item is the desired one
        disk_price_per_GB = float(disk_data[0].get('unitPrice', 0))
        logger.info(f"Disk Price per GB: {disk_price_per_GB} {currency}")

        # Extract numeric value from disk_size (e.g., '400GB' -> 400)
        disk_size_numeric_match = re.search(r'(\d+\.?\d*)', disk_size)
        if disk_size_numeric_match:
            disk_size_numeric = float(disk_size_numeric_match.group(1))
        else:
            error_msg = f"Invalid disk size format: '{disk_size}'. Expected format like '400GB'."
            logger.error(error_msg)
            return {"error": error_msg}

        disk_price = disk_price_per_GB * disk_size_numeric
        logger.info(f"Disk Price per Hour: {disk_price} {currency}")

        # Total estimated cost
        total_cost = vm_price + disk_price
        logger.info(f"Total Estimated Cost per Hour: {total_cost} {currency}")

        return {
            "vm_size": vm_size,
            "vm_price_per_hour": vm_price,
            "disk_type": disk_type,
            "disk_size": disk_size,
            "disk_price_per_GB": disk_price_per_GB,
            "disk_price_per_hour": disk_price,
            "region": region,
            "currency": currency,
            "total_estimated_cost_per_hour": total_cost
        }
    except requests.exceptions.HTTPError as http_err:
        error_msg = f"HTTP error occurred: {http_err}"
        logger.error(error_msg)
        return {"error": error_msg}
    except requests.exceptions.RequestException as req_err:
        error_msg = f"Request exception: {req_err}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"An error occurred while fetching pricing data: {e}"
        logger.error(error_msg)
        return {"error": error_msg}

# Define the tool for function calling using pydantic_function_tool
def define_tools():
    """
    Define the tools/functions that the AI model can call.

    Returns:
    - list: A list of tool definitions compatible with OpenRouter.
    """
    return [
        openai.pydantic_function_tool(GetAzurePricing, name="get_azure_pricing"),
    ]

# Function to handle AI model queries with function calling
def query_ai_model(prompt: str):
    """
    Query the AI model with function calling capability.

    Parameters:
    - prompt (str): The user-provided prompt/question.

    Returns:
    - str: The AI model's response.
    """
    try:
        tools = define_tools()

        logger.info(f"Sending prompt to AI: {prompt}")

        # Initial AI request
        response = client.chat.completions.create(
            model="openai/gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an Azure pricing assistant. Use the get_azure_pricing function to fetch pricing information when needed."},
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            tool_choice="auto",  # Allows the model to decide whether to call a tool
            parallel_tool_calls=True
        )

        message = response.choices[0].message
        logger.info(f"Received message from AI: {message.content}")

        # If the AI model decides to call a function
        if hasattr(message, "function_call") and message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)
            logger.info(f"AI requested function: {function_name} with arguments: {function_args}")

            if function_name == "get_azure_pricing":
                # Validate parameters using Pydantic
                try:
                    pricing_params = GetAzurePricing(**function_args)
                except ValidationError as ve:
                    error_msg = f"Validation error for function arguments: {ve}"
                    logger.error(error_msg)
                    return error_msg

                # Call the actual function
                pricing_result = get_azure_pricing(
                    vm_size=pricing_params.vm_size,
                    disk_type=pricing_params.disk_type,
                    disk_size=pricing_params.disk_size,
                    region=pricing_params.region
                )

                if "error" in pricing_result:
                    logger.error(f"Function call error: {pricing_result['error']}")
                    return pricing_result["error"]

                logger.info(f"Function call result: {pricing_result}")

                # Prepare the function response
                tool_response = {
                    "role": "function",
                    "name": "get_azure_pricing",
                    "content": json.dumps(pricing_result)
                }

                # Second AI request with function response
                second_response = client.chat.completions.create(
                    model="openai/gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": "You are an Azure pricing assistant."},
                        {"role": "user", "content": prompt},
                        tool_response
                    ],
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=True
                )

                final_message = second_response.choices[0].message.content
                logger.info(f"Final response from AI: {final_message}")
                return final_message

        # If no function call, return the AI's direct response
        if hasattr(message, "content") and message.content:
            logger.info(f"AI provided response: {message.content}")
            return message.content
        else:
            logger.warning("AI response content is None.")
            return "Sorry, I couldn't process your request."

    except Exception as e:
        error_msg = f"Error communicating with OpenRouter: {e}"
        logger.error(error_msg)
        return f"Sorry, I couldn't process your request. Error: {e}"

# Function to visualize VM recommendations
def visualize_recommendations(recommendations: dict):
    """
    Visualize VM recommendations using Plotly.

    Parameters:
    - recommendations (dict): Dictionary containing VM recommendations.
    """
    if not recommendations or "error" in recommendations:
        st.warning("No recommendations to visualize.")
        return

    # Convert dictionary to DataFrame
    df = pd.DataFrame([recommendations])

    # Select relevant columns
    viz_columns = ["vm_size", "disk_size", "vm_price_per_hour", "disk_price_per_hour", "total_estimated_cost_per_hour"]
    if not all(col in df.columns for col in viz_columns):
        st.error("Recommendations data is missing required fields for visualization.")
        return

    # Convert pricing columns to numeric
    df["vm_price_per_hour"] = pd.to_numeric(df["vm_price_per_hour"], errors='coerce')
    df["disk_price_per_hour"] = pd.to_numeric(df["disk_price_per_hour"], errors='coerce')
    df["total_estimated_cost_per_hour"] = pd.to_numeric(df["total_estimated_cost_per_hour"], errors='coerce')

    # Bar chart for VM and Disk Prices
    fig = px.bar(
        x=["VM Price per Hour", "Disk Price per Hour", "Total Estimated Cost per Hour"],
        y=[df["vm_price_per_hour"].iloc[0], df["disk_price_per_hour"].iloc[0], df["total_estimated_cost_per_hour"].iloc[0]],
        labels={"x": "Cost Component", "y": f"Price ({df['currency'].iloc[0]})"},
        title=f"Pricing Breakdown for {df['vm_size'].iloc[0]} with {df['disk_size'].iloc[0]} in {df['region'].iloc[0]}"
    )
    fig.update_traces(texttemplate='$%{y:.4f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig)

    # Display detailed recommendations
    st.subheader("Detailed Pricing Information")
    st.json(recommendations)

# Function to handle file uploads and generate VM recommendations
def handle_file_upload(uploaded_file):
    """
    Handle the uploaded file to extract text and generate VM recommendations.

    Parameters:
    - uploaded_file: The uploaded PDF or DOCX file.
    """
    text = extract_text_from_file(uploaded_file)
    if not text:
        st.error("Failed to extract text from the uploaded file.")
        return

    with st.spinner("Analyzing quote and generating VM recommendations..."):
        # Example prompt: Customize based on how the AI should interpret the text
        prompt = (
            f"Analyze the following quote and identify the virtual machine configurations and performance requirements. "
            f"Based on this analysis, recommend the most suitable Azure VM size, disk type, disk size, and region. "
            f"Provide the recommendations in a JSON format adhering to the following schema:\n\n"
            f"{json.dumps(GetAzurePricing.schema(), indent=2)}\n\n"
            f"Quote:\n{text}"
        )

        # Query the AI model
        recommendation = query_ai_model(prompt)

        # Display and visualize the recommendations
        if recommendation:
            try:
                recommendation_data = json.loads(recommendation)
                if "error" in recommendation_data:
                    st.error(recommendation_data["error"])
                else:
                    visualize_recommendations(recommendation_data)
            except json.JSONDecodeError:
                st.error("Failed to parse AI model's recommendation.")
        else:
            st.error("No recommendations generated.")

# Function to extract text from uploaded files
@st.cache_data
def extract_text_from_file(file):
    """
    Extract text from uploaded PDF or DOCX files.

    Parameters:
    - file: The uploaded file.

    Returns:
    - Extracted text as a string, or None if extraction fails.
    """
    try:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            st.error("Unsupported file format. Please upload a PDF or DOCX file.")
            return None
        return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Streamlit Chat Interface
def chat_interface():
    """
    Interactive chat interface with the AI assistant for real-time queries.
    """
    st.header("Chat with Azure Pricing Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "function":
                st.markdown(f"**Function Response:**\n```json\n{message['content']}\n```")
            else:
                st.markdown(message["content"])

    # Input for new message
    if prompt := st.chat_input("Ask about Azure pricing or VM recommendations"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = query_ai_model(prompt)
            if response:
                # Check if response is a JSON object (structured)
                if isinstance(response, str) and response.strip().startswith("{") and response.strip().endswith("}"):
                    try:
                        parsed_response = json.loads(response)
                        formatted_response = json.dumps(parsed_response, indent=2)
                        message_placeholder.markdown(f"```json\n{formatted_response}\n```")
                    except json.JSONDecodeError:
                        # If not a valid JSON, display as is
                        message_placeholder.markdown(response)
                else:
                    message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                message_placeholder.markdown("Sorry, I couldn't process your request.")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't process your request."})

# Streamlit Main Function
def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="Azure Pricing and VM Recommendations", layout="wide")
    st.title("Azure Pricing and VM Recommendations")
    st.markdown("""
    Welcome to the Azure Quote Helper! Upload your quotes or documents to receive VM recommendations based on your requirements.
    Additionally, you can query Azure pricing information and interact with our AI-powered assistant for real-time support.
    """)

    # File Uploader Section
    st.subheader("Upload Quote Document for VM Recommendations")
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file:
        handle_file_upload(uploaded_file)

    # Azure Pricing Lookup Section (Sidebar)
    st.sidebar.header("Azure Pricing Lookup")
    service_name = st.sidebar.selectbox("Select Azure Service Name", ["Virtual Machines"])  # Expandable for more services
    if service_name == "Virtual Machines":
        # Select Azure Region
        region_input = st.sidebar.selectbox("Select Azure Region", list(regions_mapping.keys()), index=0)
        region = regions_mapping.get(region_input, region_input.lower())

        # Fetch available VM sizes based on service and region
        vm_sizes = fetch_available_vm_sizes(service_name, region)
        if not vm_sizes:
            st.sidebar.warning("No VM sizes available for the selected service and region.")
        else:
            vm_size = st.sidebar.selectbox("Select VM Size", vm_sizes)

            # Select Disk Type
            disk_type = st.sidebar.selectbox("Select Disk Type", ["Standard HDD", "Standard SSD", "Premium SSD"])

            # Input Disk Size
            disk_size = st.sidebar.text_input("Enter Disk Size (e.g., '400GB')", value="400GB")

            # Handle Pricing Fetching
            if st.sidebar.button("Fetch Pricing"):
                if not isinstance(disk_size, str) or not re.match(r'^\d+(\.\d+)?GB$', disk_size.strip(), re.IGNORECASE):
                    st.sidebar.error("Please enter a valid disk size in the format like '400GB'.")
                else:
                    with st.spinner("Fetching pricing data..."):
                        # Call the pricing function
                        pricing_result = get_azure_pricing(vm_size, disk_type, disk_size, region)
                        if "error" in pricing_result:
                            st.sidebar.error(pricing_result["error"])
                        else:
                            st.sidebar.subheader(f"Pricing for {vm_size} with {disk_size} {disk_type} in {region_input}")
                            st.sidebar.json(pricing_result)

                            # Visualization
                            df = pd.DataFrame([pricing_result])
                            fig = px.bar(
                                x=["VM Price per Hour", "Disk Price per Hour", "Total Estimated Cost per Hour"],
                                y=[df["vm_price_per_hour"].iloc[0], df["disk_price_per_hour"].iloc[0], df["total_estimated_cost_per_hour"].iloc[0]],
                                labels={"x": "Cost Component", "y": f"Price ({df['currency'].iloc[0]})"},
                                title=f"Pricing Breakdown for {vm_size} with {disk_size} {disk_type} in {region_input}"
                            )
                            fig.update_traces(texttemplate='$%{y:.4f}', textposition='outside')
                            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                            st.sidebar.plotly_chart(fig)

    # Chat Interface Section
    chat_interface()

if __name__ == "__main__":
    main()
