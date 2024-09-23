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
from pydantic import BaseModel, ValidationError, Field
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
    vm_size: str = Field(..., description="The size of the virtual machine (e.g., 'Standard_D4s_v3').")
    disk_type: str = Field(..., description="The type of disk (e.g., 'Premium SSD'). Must be one of 'Standard HDD', 'Standard SSD', or 'Premium SSD'.")
    disk_size: int = Field(..., description="The size of the disk in GB (e.g., 400).")
    region: str = Field(..., description="The Azure region name (e.g., 'US East').")
    quantity: int = Field(1, description="The quantity of VMs.")  # Default quantity is 1

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
    'Australia East': 'australeast',
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
def get_azure_pricing(vm_size: str, disk_type: str, disk_size: int, region: str, quantity: int = 1):
    """
    Fetch pricing information for a specific VM size with disk specifications in a given region.

    Parameters:
    - vm_size (str): The size of the virtual machine (e.g., 'Standard_D4s_v3').
    - disk_type (str): The type of disk (e.g., 'Premium SSD').
    - disk_size (int): The size of the disk in GB (e.g., 400).
    - region (str): The Azure region code (e.g., 'eastus').
    - quantity (int): The quantity of VMs (default is 1).

    Returns:
    - dict: A dictionary containing VM price, Disk price, and Total estimated cost.
    """
    try:
        logger.info(f"Pricing Lookup - Service: Virtual Machines, VM Size: {vm_size}, Disk Type: {disk_type}, Disk Size: {disk_size}GB, Region: {region}, Quantity: {quantity}")
        logger.info(f"Fetching pricing for VM Size: {vm_size}, Disk Type: {disk_type}, Disk Size: {disk_size}GB, Region: {region}, Quantity: {quantity}")

        # Fetch VM pricing
        vm_filter = f"serviceName eq 'Virtual Machines' and armSkuName eq '{vm_size}' and armRegionName eq '{region}' and priceType eq 'Consumption'"
        vm_params = {
            "$filter": vm_filter,
            "$top": 1,  # Only need the first match
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

        vm_price = float(vm_data[0].get('unitPrice', 0))
        currency = vm_data[0].get('currencyCode', 'USD')
        logger.info(f"VM Price: {vm_price} {currency} per hour")

        # Fetch Disk pricing
        disk_product = disk_product_mapping.get(disk_type, disk_type)
        disk_filter = f"serviceName eq 'Storage' and productName eq '{disk_product}' and armRegionName eq '{region}' and priceType eq 'Consumption'"
        disk_params = {
            "$filter": disk_filter,
            "$top": 1,  # Only need the first match
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

        # Calculate disk price per hour based on monthly price per GB
        disk_price_per_GB_month = float(disk_data[0].get('unitPrice', 0))
        disk_price_per_hour = disk_price_per_GB_month * disk_size / (30 * 24)  # Assume 30 days in a month
        logger.info(f"Disk Price per Hour: {disk_price_per_hour} {currency}")

        # Total estimated cost
        total_cost = (vm_price + disk_price_per_hour) * quantity
        logger.info(f"Total Estimated Cost per Hour: {total_cost} {currency}")

        return {
            "vm_size": vm_size,
            "vm_price_per_hour": vm_price,
            "disk_type": disk_type,
            "disk_size": f"{disk_size}GB",
            "disk_price_per_GB_month": disk_price_per_GB_month,
            "disk_price_per_hour": disk_price_per_hour,
            "region": region,
            "currency": currency,
            "quantity": quantity,
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

# Define the OpenAPI schema for the get_azure_pricing function
get_azure_pricing_schema = {
    "name": "get_azure_pricing",
    "description": "Get the estimated hourly cost of an Azure Virtual Machine.",
    "parameters": {
        "type": "object",
        "properties": {
            "vm_size": {
                "type": "string",
                "description": "The size of the virtual machine (e.g., 'Standard_D4s_v3').",
            },
            "disk_type": {
                "type": "string",
                "description": "The type of disk (e.g., 'Premium SSD'). Must be one of 'Standard HDD', 'Standard SSD', or 'Premium SSD'.",
            },
            "disk_size": {
                "type": "integer",  # Corrected data type to integer
                "description": "The size of the disk in GB (e.g., 400).",
            },
            "region": {
                "type": "string",
                "description": "The Azure region name (e.g., 'US East').",
            },
            "quantity": {
                "type": "integer",
                "description": "The quantity of VMs (default is 1).",
            },
        },
        "required": ["vm_size", "disk_type", "disk_size", "region"],
    },
}

# Define the tool for function calling using the OpenAPI schema
def define_tools():
    """
    Define the tools/functions that the AI model can call.

    Returns:
    - list: A list of tool definitions compatible with OpenRouter.
    """
    return [
        {
            "type": "function",
            "function": get_azure_pricing_schema,
        },
    ]

# Function to handle AI model queries with function calling
def query_ai_model(prompt: str, message_history: list[dict]) -> str:
    """
    Query the AI model with function calling capability, considering message history.

    Parameters:
    - prompt (str): The user-provided prompt/question.
    - message_history (list[dict]): The history of messages in the chat.

    Returns:
    - str: The AI model's response, or pricing data as a dictionary.
    """
    try:
        tools = define_tools()

        logger.info(f"Sending prompt to AI: {prompt}")

        # Enhanced prompt to guide the model
        system_message = (
            "You are an Azure pricing assistant. Your primary function is to help users with Azure VM pricing and recommendations. "
            "You can answer user questions about Azure VMs and pricing. "
            "If a user asks for pricing, use the `get_azure_pricing` function to fetch the pricing information. "
            "Here's how to respond:\n"
            "1. **Understand the request:** Determine if the user is asking a general question or requesting pricing.\n"
            "2. **Answer general questions:** Provide helpful information about Azure VMs based on your knowledge.\n"
            "3. **Handle pricing requests:** If the user asks for pricing, follow these steps:\n"
            "    - **Identify the request:** Clearly state what pricing the user is asking for (e.g., 'You want pricing for...').\n"
            "    - **Extract the details:** Find the VM size, disk type, disk size (in GB), Azure region, and quantity.\n"
            "    - **Call the function:** Call the `get_azure_pricing` function with the extracted details.\n"
            "    - **Present the results:** Provide a clear pricing breakdown, including VM, disk, and total costs.\n"
            "4. **Be concise and informative:** Keep your responses clear, concise, and helpful."
        )

        # Add message history to the messages list
        messages = message_history + [{"role": "system", "content": system_message},
                                      {"role": "user", "content": prompt}]

        # Initial AI request
        response = client.chat.completions.create(
            model="openai/gpt-4o-2024-08-06",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Allows the model to decide whether to call a tool
        )

        response_message = response.choices[0].message

        # Handle function call if present
        if response_message.tool_calls is not None:
            # If the AI model decides to call a function
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

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
                    disk_size=int(pricing_params.disk_size),  # Convert disk size to integer
                    region=regions_mapping.get(pricing_params.region, pricing_params.region.lower()),  # Map region name
                    quantity=pricing_params.quantity
                )

                if "error" in pricing_result:
                    logger.error(f"Function call error: {pricing_result['error']}")
                    return pricing_result["error"]

                logger.info(f"Function call result: {pricing_result}")
                
                # Format the pricing result and add it to the message content
                formatted_pricing = f"**Pricing Details:**\n```json\n{json.dumps(pricing_result, indent=2)}\n```"
                response_message.content = formatted_pricing if response_message.content is None else response_message.content + "\n\n" + formatted_pricing

        # Return the AI's response, including any pricing details
        if response_message.content:
            logger.info(f"AI provided response: {response_message.content}")
            return response_message.content
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
        title=f"Pricing Breakdown for {df['quantity'].iloc[0]} x {df['vm_size'].iloc[0]} with {df['disk_size'].iloc[0]} in {df['region'].iloc[0]}"
    )
    fig.update_traces(texttemplate='$%{y:.4f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig)

    # Display detailed recommendations
    st.subheader("Detailed Pricing Information")
    st.json(recommendations)

# Function to handle file uploads and generate VM recommendations
def handle_file_upload(uploaded_file, message_history):
    """
    Handle the uploaded file to extract text and generate VM recommendations.

    Parameters:
    - uploaded_file: The uploaded PDF or DOCX file.
    - message_history (list[dict]): The history of messages in the chat.
    """
    text = extract_text_from_file(uploaded_file)
    if not text:
        st.error("Failed to extract text from the uploaded file.")
        return

    with st.spinner("Analyzing document and generating VM recommendations..."):
        # Example prompt: Customize based on how the AI should interpret the text
        prompt = (
            f"Analyze the following document, focusing on technical specifications and requirements. "
            f"Identify details relevant to an Azure VM migration, such as:\n"
            "- CPU cores and speed\n"
            "- RAM\n"
            "- Storage capacity and type (HDD, SSD)\n"
            "- Network bandwidth\n"
            "- Application requirements (e.g., database, web server)\n"
            "- Expected workload and usage patterns\n\n"
            f"Based on your analysis, recommend the most suitable Azure VM size, disk type, disk size (in GB), and region. "
            f"Extract all necessary parameters for the 'get_azure_pricing' function and call it to "
            f"include the pricing information in your response.  Format your final response as a JSON object "
            f"that includes both the recommendations and pricing details.\n\n"
            f"Document:\n{text}"
        )

        # Query the AI model
        recommendations = query_ai_model(prompt, message_history)

        # Display and visualize the recommendations
        if recommendations:
            try:
                recommendations_data = json.loads(recommendations) if isinstance(recommendations, str) else recommendations
                if "error" in recommendations_data:
                    st.error(recommendations_data["error"])
                else:
                    visualize_recommendations(recommendations_data)
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

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create two columns
    col1, col2 = st.columns(2)

    # Azure Pricing Lookup Section (Column 1)
    with col1:
        st.header("Azure Pricing Lookup")
        service_name = st.selectbox("Select Azure Service Name", ["Virtual Machines"])  # Expandable for more services
        if service_name == "Virtual Machines":
            # Select Azure Region
            region_input = st.selectbox("Select Azure Region", list(regions_mapping.keys()), index=0)
            region = regions_mapping.get(region_input, region_input.lower())

            # Fetch available VM sizes based on service and region
            vm_sizes = fetch_available_vm_sizes(service_name, region)
            if not vm_sizes:
                st.warning("No VM sizes available for the selected service and region.")
            else:
                vm_size = st.selectbox("Select VM Size", vm_sizes)

                # Select Disk Type
                disk_type = st.selectbox("Select Disk Type", list(disk_product_mapping.keys()))

                # Input Disk Size
                disk_size = st.number_input("Enter Disk Size (GB)", min_value=1, value=400)

                # Input Quantity
                quantity = st.number_input("Quantity", min_value=1, value=1)

                # Handle Pricing Fetching
                if st.button("Fetch Pricing"):
                    with st.spinner("Fetching pricing data..."):
                        # Call the pricing function
                        pricing_result = get_azure_pricing(vm_size, disk_type, disk_size, region, quantity)
                        if "error" in pricing_result:
                            st.error(pricing_result["error"])
                        else:
                            st.subheader(f"Pricing for {quantity} x {vm_size} with {disk_size}GB {disk_type} in {region_input}")
                            st.json(pricing_result)

                            # Visualization
                            df = pd.DataFrame([pricing_result])
                            fig = px.bar(
                                x=["VM Price per Hour", "Disk Price per Hour", "Total Estimated Cost per Hour"],
                                y=[df["vm_price_per_hour"].iloc[0], df["disk_price_per_hour"].iloc[0], df["total_estimated_cost_per_hour"].iloc[0]],
                                labels={"x": "Cost Component", "y": f"Price ({df['currency'].iloc[0]})"},
                                title=f"Pricing Breakdown for {quantity} x {vm_size} with {disk_size}GB {disk_type} in {region_input}"
                            )
                            fig.update_traces(texttemplate='$%{y:.4f}', textposition='outside')
                            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                            st.plotly_chart(fig)

                # Handle Add to Chat
                if st.button("Add to Chat"):
                    with st.spinner("Fetching pricing data and adding to chat..."):
                        pricing_result = get_azure_pricing(vm_size, disk_type, disk_size, region, quantity)
                        if "error" in pricing_result:
                            st.error(pricing_result["error"])
                        else:
                            # Add pricing data to chat history
                            st.session_state.messages.append({"role": "assistant", "content": pricing_result})

    # AI Chat Interface (Column 2)
    with col2:
        st.header("Chat with Azure Pricing Assistant")
        
        # Create a container with a scrollbar for the chat history
        chat_container = st.container()
        with chat_container:
            # Display existing messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "function":
                        st.markdown(f"**Function Response:**\n```json\n{message['content']}\n```")
                    else:
                        st.markdown(message["content"])

        # Chat input outside the chat history container
        with st.container():
            if prompt := st.chat_input("Ask about Azure pricing or VM recommendations"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:  # Add new messages inside the chat_container
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        response = query_ai_model(prompt, st.session_state.messages)
                        if response:
                            if isinstance(response, dict):  # Check if response is a dictionary (pricing data)
                                # Format and display pricing data
                                message_placeholder.markdown(f"**Pricing Details:**\n```json\n{json.dumps(response, indent=2)}\n```")
                            else:  # Assume it's a regular text response
                                message_placeholder.markdown(response)
                            if response is not None:  # Only append if response is not None
                                st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            message_placeholder.markdown("Sorry, I couldn't process your request.")
                            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't process your request."})

    # File Uploader Section
    st.subheader("Upload Document for VM Recommendations")
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file:
        handle_file_upload(uploaded_file, st.session_state.messages)

if __name__ == "__main__":
    main()
