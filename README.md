# BigQuery Natural Language Query Tool

This application allows users to query BigQuery datasets using natural language. It leverages Google's Gemini model to translate natural language queries into SQL, execute them on BigQuery, and display the results with visualization options.

## Features

- **User Authentication**: Secure login system with password hashing
- **Natural Language to SQL**: Convert plain English questions into SQL queries using Google's Gemini AI model
- **BigQuery Integration**: Execute generated SQL queries directly on BigQuery
- **Interactive UI**: Clean Streamlit interface with query history, schema viewing, and result pagination
- **Dynamic Visualizations**: Automatically suggests and creates appropriate data visualizations
- **Customizable Charts**: Modify visualization types, titles, and parameters
- **Query History**: Track and review previous queries and their results
- **Pagination**: Handle large result sets with built-in pagination

## Setup Instructions

### Prerequisites

1. Google Cloud Platform account with BigQuery access
2. Service account key with BigQuery permissions
3. Google AI Platform API key for Gemini access (gemini-1.5-flash-002 model)
4. Python 3.8 or higher

### Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.streamlit/secrets.toml` file with the following structure:

```toml
[authentication]
email = "your_login_email"
password_hash = "hashed_password"
gemini = "your_gemini_api_key"

[gcp_service_account]
type = "service_account"
project_id = "your_project_id"
private_key_id = "your_private_key_id"
private_key = "your_private_key"
client_email = "your_client_email"
client_id = "your_client_id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your_client_cert_url"
dataset_id = "your_dataset_id"
```

You can use the included `auth.py` script to generate the password hash.

### Running the Application

```bash
cd path/to/your/app
streamlit run main.py
```

## Usage

1. Log in with your credentials
2. The application automatically connects to your BigQuery dataset
3. Explore available tables and their schemas in the expandable section
4. Enter a natural language query in the text area
5. Click "Execute Query" to convert your query to SQL, run it, and see results
6. Click "Plot" to generate an appropriate visualization for your query results
7. Customize the visualization using the sidebar options
8. Review your query history in the expandable section

## Example Queries

- "Show me the top 10 customers by revenue"
- "What were our sales last month by product category?"
- "Calculate the average order value by region for Q1 2023"
- "Find customers who haven't placed an order in the last 30 days"

## Visualization Types

The application supports various visualization types:
- Bar charts (vertical or horizontal)
- Line charts
- Pie charts
- Scatter plots
- Histograms
- Box plots
- Heatmaps
- Area charts

## Security Considerations

- Credentials are stored securely in the Streamlit secrets management system
- Passwords are hashed using SHA-256
- Authentication is required to access the application
- Service account keys should have the minimum necessary permissions
- Consider implementing additional security measures for production environments

## Development

Built with:
- Streamlit for the web interface
- Google BigQuery for data storage and querying
- Google Gemini AI for natural language processing
- Plotly for interactive data visualizations
- Pandas for data manipulation
