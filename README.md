# Azure Quote Helper

Azure Quote Helper is a Streamlit-based application that assists in analyzing Azure VM quotes, providing pricing information, and recommending suitable VM sizes and SKUs.

## Features

1. **File Upload**: Upload PDF or DOCX files containing Azure VM quotes for analysis.
2. **Quote Analysis**: Analyze uploaded quotes and receive recommendations for suitable Azure VM sizes and SKUs.
3. **Azure Pricing Lookup**: Fetch current pricing details for specific Azure services.
4. **Chat Interface**: Interact with an AI assistant to ask questions about Azure pricing and VM recommendations.
5. **Data Visualization**: View recommendations in a tabular format and export to CSV.

## Installation

1. Clone the repository:
   git clone https://github.com/paulhshort/AzureQuoteHelper.git
   cd AzureQuoteHelper

2. Install the required dependencies:
   pip install -r requirements.txt

3. Set up your environment variables (see Configuration section).

## Configuration

Create a `.env` file in the root directory of the project and add the following environment variables:
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here


Replace `your_openai_api_key_here` with your actual OpenAI API key, and `your_openrouter_api_key_here` with your OpenRouter API key if you're using it.

## Usage

1. Run the Streamlit app:
   streamlit run AzureQuoteHelper.py


2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the app:
- Upload a PDF or DOCX file containing an Azure VM quote.
- The app will analyze the quote and provide VM recommendations.
- Use the chat interface to ask questions about Azure pricing or VM recommendations.
- View and export recommendations in tabular format.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
