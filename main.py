import os
import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import google.generativeai as genai
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional, List
import hashlib

# Configuration and setup
st.set_page_config(
    page_title="Natural Language BigQuery",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'total_pages' not in st.session_state:
    st.session_state.total_pages = 0
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'current_sql' not in st.session_state:
    st.session_state.current_sql = None
if 'execution_error' not in st.session_state:
    st.session_state.execution_error = None
if 'visualization_type' not in st.session_state:
    st.session_state.visualization_type = None
if 'visualization_config' not in st.session_state:
    st.session_state.visualization_config = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

class BigQueryNLInterface:
    def __init__(self, credentials, project_id: str, gemini_api_key: str):
        """Initialize the BigQuery Natural Language Interface."""
        self.project_id = project_id
        self.credentials = credentials
        self.bq_client = bigquery.Client(credentials=self.credentials, project=project_id)
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-002')

    def get_dataset_schema(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch schema information for all tables in a BigQuery dataset."""
        dataset_ref = self.bq_client.dataset(dataset_id)
        tables = list(self.bq_client.list_tables(dataset_ref))
        schema_info = {}
        for table in tables:
            table_ref = self.bq_client.get_table(table.reference)
            schema_info[table.table_id] = {
                "description": table_ref.description,
                "columns": [{"name": field.name, "type": field.field_type, "description": field.description} for field in table_ref.schema]
            }
        return schema_info

    def generate_sql_query(self, user_query: str, dataset_id: str, schema_info: Dict[str, Any], query_history: List[Dict[str, Any]]) -> str:
        """Generate a SQL query based on the user's natural language input using Gemini API, including query history."""
        schema_description = json.dumps(schema_info, indent=2)
        history_text = "\n".join([f"User Query: {item['user_query']}\nSQL Query: {item['sql_query']}" for item in query_history])
        prompt = f"""You are an expert SQL query generator for Google BigQuery. Based on the following dataset schema: ```{schema_description}``` And the following query history: ```{history_text}``` Convert the following natural language query into a valid SQL query for BigQuery: "{user_query}" Dataset ID: {dataset_id} Project ID: {self.project_id} Return ONLY the SQL query without any explanations or markdown formatting. The query should be optimized and include proper table references with project and dataset."""
        response = self.model.generate_content(prompt)
        generated_sql = response.text.strip()
        if generated_sql.startswith('```sql'):
            generated_sql = generated_sql.replace('```sql', '').replace('```', '').strip()
        elif generated_sql.startswith('```'):
            generated_sql = generated_sql.replace('```', '').strip()
        return generated_sql

    def execute_query(self, sql_query: str, page_size: int = 100, page: int = 0) -> Tuple[Optional[pd.DataFrame], int, Optional[str]]:
        """Execute a SQL query on BigQuery and handle pagination of results."""
        try:
            job_config = bigquery.QueryJobConfig(allow_large_results=True)
            query_job = self.bq_client.query(sql_query, job_config=job_config)
            results = list(query_job.result())
            total_rows = len(results)
            total_pages = (total_rows + page_size - 1) // page_size if total_rows > 0 else 0
            if total_rows > 0:
                field_names = [field.name for field in query_job.result().schema]
                start_index = page * page_size
                end_index = min(start_index + page_size, total_rows)
                page_results = results[start_index:end_index]
                rows = []
                for row in page_results:
                    row_dict = {}
                    for i, value in enumerate(row):
                        row_dict[field_names[i]] = value
                    rows.append(row_dict)
                result_df = pd.DataFrame(rows)
            else:
                result_df = pd.DataFrame()
            return result_df, total_pages, None
        except Exception as e:
            error_msg = str(e)
            print(f"Error executing query: {error_msg}")
            return None, 0, error_msg

    def suggest_visualization(self, df: pd.DataFrame, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """Use Gemini to suggest the best visualization type and configuration."""
        df_columns = list(df.columns)
        df_sample = df.head(5).to_json(orient="records")
        df_dtypes = {col: str(df[col].dtype) for col in df_columns}
        prompt = f"""You are an expert data visualization specialist. I have a dataset with the following columns: {df_columns} The data types for these columns are: {df_dtypes} Here's a sample of the data: {df_sample} This data is from a query that answers: "{user_query}" Please suggest the most appropriate visualization type and configuration. The available chart types are: bar, line, pie, scatter, histogram, box, heatmap, area. Respond in JSON format with the following structure: {{"chart_type": "the recommended chart type", "title": "suggested chart title", "x_axis": "column name for x-axis", "y_axis": "column name for y-axis or list of columns for multiple series", "color": "column name for color differentiation (if applicable)", "orientation": "horizontal or vertical (for bar charts)", "reasoning": "brief explanation for your recommendation"}} Return ONLY the JSON without any additional text or formatting."""
        try:
            response = self.model.generate_content(prompt)
            suggestion_text = response.text.strip()
            if "```json" in suggestion_text:
                suggestion_text = suggestion_text.split("```json")[1].split("```")[0].strip()
            elif "```" in suggestion_text:
                suggestion_text = suggestion_text.split("```")[1].split("```")[0].strip()
            suggestion = json.loads(suggestion_text)
            return suggestion["chart_type"], suggestion
        except Exception as e:
            print(f"Error suggesting visualization: {str(e)}")
            return "bar", {"chart_type": "bar", "title": "Query Results", "x_axis": df_columns[0], "y_axis": df_columns[1] if len(df_columns) > 1 else df_columns[0], "orientation": "vertical"}

def create_visualization(df: pd.DataFrame, viz_type: str, viz_config: Dict[str, Any]):
    """Create and return a Plotly visualization based on the provided configuration."""
    title = viz_config.get("title", "Query Results")
    x_column = viz_config.get("x_axis")
    y_column = viz_config.get("y_axis")
    color_column = viz_config.get("color")
    orientation = viz_config.get("orientation", "vertical")
    if x_column and x_column not in df.columns:
        x_column = df.columns[0]
    if isinstance(y_column, list):
        valid_y_columns = [col for col in y_column if col in df.columns]
        y_column = valid_y_columns if valid_y_columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    elif y_column and y_column not in df.columns:
        y_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if color_column and color_column not in df.columns:
        color_column = None
    if viz_type == "bar":
        if orientation == "horizontal":
            fig = px.bar(df, y=x_column, x=y_column, color=color_column, title=title, orientation='h')
        else:
            fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
    elif viz_type == "line":
        if isinstance(y_column, list):
            fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
        else:
            fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
    elif viz_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
    elif viz_type == "pie":
        fig = px.pie(df, values=y_column, names=x_column, title=title)
    elif viz_type == "histogram":
        fig = px.histogram(df, x=x_column, title=title)
    elif viz_type == "box":
        fig = px.box(df, x=x_column, y=y_column, color=color_column, title=title)
    elif viz_type == "heatmap":
        pivot_df = df.pivot(index=x_column, columns=y_column, values=color_column)
        fig = px.imshow(pivot_df, title=title, labels=dict(x=y_column, y=x_column, color=color_column))
    elif viz_type == "area":
        fig = px.area(df, x=x_column, y=y_column, color=color_column, title=title)
    else:
        fig = px.bar(df, x=x_column, y=y_column, title=title)
    fig.update_traces(hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>')
    fig.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'), margin=dict(l=40, r=40, t=60, b=40))
    return fig

def execute_current_query():
    """Execute the current SQL query stored in session state and handle pagination."""
    if not st.session_state.current_sql:
        return
    with st.spinner("Executing query..."):
        interface = BigQueryNLInterface(
            st.session_state.dataset_info["credentials"], 
            st.session_state.dataset_info["project_id"], 
            st.session_state.dataset_info["gemini_api_key"]
        )
        results_df, total_pages, error = interface.execute_query(st.session_state.current_sql, page=st.session_state.current_page)
        st.session_state.results_df = results_df
        st.session_state.total_pages = total_pages
        st.session_state.execution_error = error
        st.session_state.visualization_type = None
        st.session_state.visualization_config = None
        current_query_index = len(st.session_state.query_history) - 1
        if current_query_index >= 0:
            if error:
                st.session_state.query_history[current_query_index]["error"] = error
            else:
                st.session_state.query_history[current_query_index]["results"] = results_df

def go_to_previous_page():
    """Navigate to the previous page of query results, if available."""
    if st.session_state.current_page > 0:
        st.session_state.current_page -= 1
        execute_current_query()

def go_to_next_page():
    """Navigate to the next page of query results, if available."""
    if st.session_state.current_page < st.session_state.total_pages - 1:
        st.session_state.current_page += 1
        execute_current_query()

def suggest_visualization():
    """Use the BigQueryNLInterface to suggest a visualization for the current results."""
    if st.session_state.results_df is None or st.session_state.results_df.empty:
        st.warning("No data available to visualize.")
        return
    current_query_index = len(st.session_state.query_history) - 1
    if current_query_index >= 0:
        user_query = st.session_state.query_history[current_query_index]["user_query"]
    else:
        user_query = "Query results"
    interface = BigQueryNLInterface(
        st.session_state.dataset_info["credentials"], 
        st.session_state.dataset_info["project_id"], 
        st.session_state.dataset_info["gemini_api_key"]
    )
    with st.spinner("Analyzing data and generating visualization..."):
        viz_type, viz_config = interface.suggest_visualization(st.session_state.results_df, user_query)
        st.session_state.visualization_type = viz_type
        st.session_state.visualization_config = viz_config

def authenticate_user(email, password):
    """Authenticate the user with provided email and password using secure hash comparison."""
    try:
        # Get credentials from Streamlit secrets
        auth_email = st.secrets["authentication"]["email"]
        stored_password_hash = st.secrets["authentication"]["password_hash"]
        
        # Hash the provided password
        provided_password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Compare email and password hash
        if email == auth_email and provided_password_hash == stored_password_hash:
            st.session_state.authenticated = True
            return True
        return False
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False

def logout():
    """Log out the user by resetting session state."""
    st.session_state.authenticated = False
    st.session_state.dataset_info = None

def connect_to_bigquery():
    """Establish connection to BigQuery using credentials from secrets."""
    try:
        # Get GCP credentials from secrets
        credentials_info = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        
        # Get other configuration
        project_id = credentials_info["project_id"]
        gemini_api_key = st.secrets["authentication"]["gemini"]
        
        # Get dataset_id - first try from secrets, then fallback to a default
        dataset_id = st.secrets.get("dataset_id", "")
        if not dataset_id:
            # Check if dataset_id is in gcp_service_account custom field
            dataset_id = credentials_info.get("dataset_id", "")
            if not dataset_id:
                st.error("Dataset ID not found in secrets.")
                return False, "Dataset ID not found in secrets."
        
        # Create interface with credentials from secrets
        interface = BigQueryNLInterface(credentials, project_id, gemini_api_key)
        dataset_schema = interface.get_dataset_schema(dataset_id)
        
        if dataset_schema:
            st.session_state.dataset_info = {
                "credentials": credentials, 
                "project_id": project_id, 
                "gemini_api_key": gemini_api_key, 
                "dataset_id": dataset_id, 
                "schema": dataset_schema
            }
            return True, "Successfully connected to BigQuery!"
        else:
            return False, "No tables found in the dataset."
    except Exception as e:
        return False, f"Connection error: {str(e)}"

# Login UI
def show_login_page():
    """Display the login form."""
    st.title("Natural Language Tool - Login")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.write("Please log in to access the application")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if authenticate_user(email, password):
                    # Try to connect to BigQuery automatically
                    success, message = connect_to_bigquery()
                    if success:
                        st.success("Login successful! Connected to BigQuery.")
                        st.rerun()
                    else:
                        st.error(f"Login successful, but couldn't connect to BigQuery: {message}")
                        st.session_state.authenticated = True
                        st.rerun()
                else:
                    st.error("Invalid email or password")

# Main application UI
def show_application():
    """Display the main application UI after successful authentication."""
    st.title("Natural Language Tool")
    
    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()
    
    if not st.session_state.dataset_info:
        st.info("Connecting to BigQuery...")
        success, message = connect_to_bigquery()
        if success:
            st.success(message)
            st.experimental_rerun()
        else:
            st.error(message)
            st.warning("Please check if the secrets file contains all required information.")
            return
    
    # Visualization settings sidebar
    st.sidebar.title("Visualization Settings")
    st.sidebar.info("Click 'Plot' after query execution to automatically suggest a visualization.")
    if st.session_state.visualization_type and st.session_state.visualization_config:
        st.sidebar.subheader("Customize Visualization")
        chart_types = ["bar", "line", "pie", "scatter", "histogram", "box", "heatmap", "area"]
        selected_type = st.sidebar.selectbox("Chart Type", chart_types, index=chart_types.index(st.session_state.visualization_type) if st.session_state.visualization_type in chart_types else 0)
        if selected_type != st.session_state.visualization_type:
            st.session_state.visualization_type = selected_type
        custom_title = st.sidebar.text_input("Chart Title", value=st.session_state.visualization_config.get("title", "Query Results"))
        if custom_title != st.session_state.visualization_config.get("title"):
            st.session_state.visualization_config["title"] = custom_title
        if st.session_state.results_df is not None and not st.session_state.results_df.empty:
            available_columns = list(st.session_state.results_df.columns)
            x_column = st.sidebar.selectbox("X-axis", available_columns, index=available_columns.index(st.session_state.visualization_config.get("x_axis")) if st.session_state.visualization_config.get("x_axis") in available_columns else 0)
            if x_column != st.session_state.visualization_config.get("x_axis"):
                st.session_state.visualization_config["x_axis"] = x_column
            if selected_type in ["bar", "line", "scatter", "box", "area"]:
                y_columns = st.sidebar.multiselect("Y-axis", available_columns, default=[st.session_state.visualization_config.get("y_axis")] if st.session_state.visualization_config.get("y_axis") in available_columns else [available_columns[1]])
                if y_columns:
                    st.session_state.visualization_config["y_axis"] = y_columns
            if selected_type in ["bar", "line", "scatter", "box", "area"]:
                color_options = ["None"] + available_columns
                current_color = st.session_state.visualization_config.get("color", "None")
                if current_color not in color_options:
                    current_color = "None"
                color_column = st.sidebar.selectbox("Color By", color_options, index=color_options.index(current_color))
                st.session_state.visualization_config["color"] = None if color_column == "None" else color_column
            if selected_type == "bar":
                orientation = st.sidebar.radio("Orientation", ["vertical", "horizontal"], index=0 if st.session_state.visualization_config.get("orientation", "vertical") == "vertical" else 1)
                st.session_state.visualization_config["orientation"] = orientation
    
    # Main content area
    with st.expander("Available Tables and Schemas", expanded=False):
        for table_name, table_info in st.session_state.dataset_info["schema"].items():
            st.subheader(table_name)
            if table_info.get("description"):
                st.write(f"**Description:** {table_info['description']}")
            schema_df = pd.DataFrame(table_info["columns"])
            st.dataframe(schema_df)
    
    user_query = st.text_area("Enter your question in natural language:", placeholder="Example: Show me the top 5 topics for the last month")

    col1, col2 = st.columns([1, 9])
    with col1:
        execute_button = st.button("Execute Query")

    if execute_button and user_query:
        try:
            interface = BigQueryNLInterface(
                st.session_state.dataset_info["credentials"], 
                st.session_state.dataset_info["project_id"], 
                st.session_state.dataset_info["gemini_api_key"]
            )
            sql_query = interface.generate_sql_query(user_query, st.session_state.dataset_info["dataset_id"], st.session_state.dataset_info["schema"], st.session_state.query_history)
            st.session_state.query_history.append({"user_query": user_query, "sql_query": sql_query, "results": None, "error": None})
            st.session_state.current_sql = sql_query
            st.session_state.results_df = None
            st.session_state.execution_error = None
            st.session_state.current_page = 0
            st.session_state.visualization_type = None
            st.session_state.visualization_config = None
            execute_current_query() # Execute the query after generating
        except Exception as e:
            st.error(f"Error generating SQL: {str(e)}")

    if st.session_state.execution_error:
        st.error("This query can not be executed, please try again later.")
    elif st.session_state.results_df is not None:
        st.subheader("Query Results")
        st.dataframe(st.session_state.results_df)
        if st.session_state.total_pages > 1:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("Previous Page"):
                    go_to_previous_page()
            with col2:
                st.write(f"Page {st.session_state.current_page + 1} of {st.session_state.total_pages}")
            with col3:
                if st.button("Next Page"):
                    go_to_next_page()
        col1, col2 = st.columns([1, 9])
        with col1:
            plot_button = st.button("Plot")
        if plot_button:
            suggest_visualization()
        if st.session_state.visualization_type and st.session_state.visualization_config:
            st.subheader("Data Visualization")
            if "reasoning" in st.session_state.visualization_config:
                st.info(f"**Why this visualization?** {st.session_state.visualization_config['reasoning']}")
            fig = create_visualization(st.session_state.results_df, st.session_state.visualization_type, st.session_state.visualization_config)
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Query History", expanded=False):
        if st.session_state.query_history:
            for i, query_item in enumerate(reversed(st.session_state.query_history)):
                st.write(f"**Query {len(st.session_state.query_history) - i}:** {query_item['user_query']}")
                if query_item.get("error"):
                    st.error(f"Error: {query_item['error']}")
                elif query_item.get("results") is not None:
                    st.dataframe(query_item['results'])
                st.divider()
        else:
            st.write("No queries yet.")

# Main app flow
if st.session_state.authenticated:
    show_application()
else:
    show_login_page()

st.markdown("---")
st.caption("Raven Data's Natural Language Query Tool - Powered by Google Gemini")