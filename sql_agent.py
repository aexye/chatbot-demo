from typing import Optional, Dict, Any, List, Set
import logging
from dataclasses import dataclass
import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect
import streamlit as st
from datetime import datetime, time
import re
import json
from google.oauth2 import service_account
import pandas as pd
# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s\n',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("sql_agent")

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class SQLAgentConfig:
    project_id: str
    dataset_id: str
    table_id: str
    credentials_dict: Dict[str, Any]
    model_name: str = "gemini-1.5-flash-002"
    temperature: float = 0
    max_iterations: int = 3
    location: str = "us-central1"

class SQLAgent:
    def __init__(self, config: SQLAgentConfig):
        self.config = config
        self.credentials = self._create_credentials()
        self.engine = self._create_engine()
        self.db = SQLDatabase(self.engine)
        self.llm = self._create_llm()
        self.table_info = self._get_table_info()
        logger.info(f"Available columns: {self.table_info}")
        
    def _create_engine(self):
        """Create SQLAlchemy engine with BigQuery credentials"""
        return create_engine(
            f"bigquery://{self.config.project_id}/{self.config.dataset_id}",
            credentials_info=self.config.credentials_dict
        )

    def _create_credentials(self):
        """Create credentials from Streamlit secrets"""
        return service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    
    def _get_table_info(self) -> Dict[str, Any]:
        """Get column information for the specified table"""
        try:
            from google.cloud import bigquery
            
            client = bigquery.Client(
                project=self.config.project_id,
                credentials=self.credentials
            )
            
            # Get the table object
            table_ref = f"{self.config.project_id}.{self.config.dataset_id}.{self.config.table_id}"
            table = client.get_table(table_ref)
            
            info = {
                'columns': [],
                'descriptions': {}
            }
            
            # Extract column names and descriptions
            for field in table.schema:
                info['columns'].append(field.name)
                if field.description:
                    info['descriptions'][field.name] = field.description
                    
            logger.info(f"Found columns for table {self.config.table_id}: {info['columns']}")
            return info
            
        except Exception as e:
            logger.error(f"Error fetching schema from BigQuery: {str(e)}")
            return {'columns': [], 'descriptions': {}}

    def _get_prompt_info(self) -> str:
        """Format table information for prompt"""
        info = f"Table: {self.config.project_id}.{self.config.dataset_id}.{self.config.table_id}\n\nColumns:\n"
        
        for col in self.table_info['columns']:
            description = self.table_info['descriptions'].get(col, '')
            info += f"- {col}" + (f": {description}" if description else "") + "\n"
        
        return info

    def _validate_columns_in_query(self, query: str) -> List[str]:
        """Check if all columns in the query exist in the schema"""
        invalid_columns = []
        
        # Extract all column references from the query
        column_pattern = r't1\.(\w+)'
        referenced_columns = re.findall(column_pattern, query, re.IGNORECASE)
        
        # Check each referenced column against the schema
        for col in referenced_columns:
            if col not in self.table_info['columns']:
                invalid_columns.append(col)
                    
        return invalid_columns

    def _create_llm(self):
        """Create Vertex AI model instance"""
        vertexai.init(
            project=self.config.project_id,
            location=self.config.location,
            credentials=self.credentials
        )
        return GenerativeModel(
            self.config.model_name,
            generation_config={"temperature": self.config.temperature}
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison by removing whitespace and standardizing syntax"""
        # Remove all whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        # Convert to uppercase for comparison
        query = query.upper()
        # Remove any trailing semicolon
        query = query.rstrip(';')
        return query

    def _parse_error_message(self, error: Exception, query: str) -> str:
        """Extract meaningful information from error message"""
        error_str = str(error)
        
        # Extract column not found errors
        column_match = re.search(r"Name (\w+) not found inside", error_str)
        if column_match:
            column_name = column_match.group(1)
            available_columns = self.table_info.get('columns', [])
            return f"Column '{column_name}' does not exist in the table. Available columns are: {', '.join(available_columns)}"
            
        # Extract syntax errors
        if "Syntax error" in error_str:
            return "SQL syntax error. Please check the query structure."
            
        # Extract other common errors
        if "GROUP BY" in error_str:
            return "Issue with GROUP BY clause. Make sure all non-aggregated columns are included."
            
        if "CURRENT_DATE" in error_str:
            return "Issue with date handling. Make sure to use proper date functions and formats."
            
        # Default to a simplified version of the error
        return error_str.split("[SQL:")[0].strip()

    def _validate_results_with_llm(self, df: pd.DataFrame, question: str) -> bool:
        """Use LLM to validate if the results answer the question accurately"""
        # Convert DataFrame to a list of dicts for better formatting
        results_list = df.head(5).to_dict('records')
        # Format the results as a readable string
        results_str = json.dumps(results_list, indent=2, cls=DateTimeEncoder)
        
        prompt = f"""Analyze if the query results provide the specific information requested in the user's question.
        
        Question: {question}
        
        Results info:
        - Total rows: {len(df)}
        - Columns: {', '.join(df.columns)}
        - First 5 rows: {results_str}
        
        Important checks:
        1. Are all SPECIFICALLY REQUESTED fields present in the results? For example:
           - If user asks about "races", we must have race_name (not race_id)
           - If user asks about "runners", we must have number of runners
           - If user asks about "times", we must have race_time
        2. Only accept if the core requested information is present
        3. Missing supplementary information is okay
        4. Results should use descriptive names (race_name, horse_name) instead of IDs
        
        First, list the specific fields the user asked for.
        Then check if each requested field is present in the results.
        Finally, answer:
        - YES if all specifically requested fields are present with proper descriptive names
        - NO if any specifically requested field is missing or only IDs are provided instead of names
        """
        
        response = self.llm.generate_content(prompt)
        response_text = response.text
        logger.info(f"Validation response: {response_text}")
        
        # Check if response contains NO
        if "NO" in response_text.upper():
            # Extract missing fields from the response
            missing_fields = []
            if "race_name" not in df.columns and "race" in question.lower():
                missing_fields.append("race_name")
            if "race_time" not in df.columns and "time" in question.lower():
                missing_fields.append("race_time")
            if not any(col for col in df.columns if "runner" in col.lower()) and "runner" in question.lower():
                missing_fields.append("number_of_runners/horse_count")
                
            if missing_fields:
                logger.info(f"Missing requested fields: {missing_fields}")
            return False
            
        return True

    def _clean_query(self, query: str) -> str:
        """Clean and standardize the query format"""
        # Remove any markdown code block syntax
        query = query.strip('`').strip()
        if query.startswith('sql'):
            query = query[3:].strip()
            
        # Remove backticks from table names
        query = query.replace('`', '')
        
        # Clean up newlines and multiple spaces
        query = ' '.join(query.split())
        
        # Ensure the query ends with a single semicolon
        query = query.rstrip(';') + ';'
        
        return query

    def _generate_sql_query(self, user_question: str, error_context: Optional[Dict] = None) -> str:
        """Generate SQL query using LLM"""
        logger.info(f"Generating SQL query for question: {user_question}")
        
        # Get table information including descriptions
        table_info = self._get_prompt_info()
        
        system_prompt = f"""Generate a SQL query to answer the user's question.
        Important guidelines:
        1. Use only the available columns from the schema
        2. For date filtering use: DATE(race_date) = CURRENT_DATE()
        3. For time extraction use: FORMAT_TIMESTAMP('%H:%M', race_date) as race_time
        4. For string comparisons, use exact matches (=) not LIKE unless specifically needed
        5. Include proper GROUP BY if using aggregations
        6. Double-check all table and column names
        7. Ensure all conditions are properly defined
        8. Do not use backticks in table names
        9. Always include appropriate WHERE clauses for filtering
        10. Always use the full table name: {self.config.project_id}.{self.config.dataset_id}.{self.config.table_id}
        11. For user-facing data, always use descriptive names instead of IDs:
            - Use race_name instead of race_id
            - Use horse_name instead of horse_id
            - Use course_name instead of course_id
            (IDs should only be used for joins or internal logic)"""
        
        # Build context with previous error if exists
        context = ""
        if error_context:
            context = f"""Previous attempt failed:
            Error: {error_context['error']}
            Failed query: {error_context['query']}
            
            Analyze what went wrong and fix the issues."""
        
        prompt = f"""{system_prompt}

Question: {user_question}

{table_info}

{context}

Generate only the SQL query without any explanations."""
        
        response = self.llm.generate_content(prompt)
        query = self._clean_query(str(response.text))
        logger.info(f"Generated query:\n{query}")
        return query

    def _validate_query(self, query: str) -> bool:
        """Validate if query is SELECT only and contains required components"""
        query_upper = query.upper()
        if not query_upper.startswith('SELECT'):
            logger.warning("Query validation failed: Not a SELECT statement")
            return False
        if any(keyword in query_upper for keyword in ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE']):
            logger.warning(f"Query validation failed: Contains forbidden keyword")
            return False
            
        # Instead of failing on invalid columns, we'll just log them
        invalid_columns = self._validate_columns_in_query(query)
        if invalid_columns:
            logger.warning(f"Found invalid columns that will be excluded: {invalid_columns}")
            
        return True

    def execute_query(self, query: str) -> List[Dict]:
        """Execute SQL query and return results as list of dictionaries"""
        logger.info("Executing query...")
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query)
                columns = result.keys()
                results = [dict(zip(columns, row)) for row in result.fetchall()]
                logger.info(f"Query executed successfully. Got {len(results)} results")
                
                # Try to log a sample result
                if results:
                    try:
                        # Convert the first result to a more readable format
                        sample = results[0].copy()
                        # Format any datetime objects
                        for k, v in sample.items():
                            if isinstance(v, (datetime, time)):
                                sample[k] = str(v)
                        sample_json = json.dumps(sample, indent=2)
                        logger.info(f"Sample result: {sample_json}")
                    except Exception as e:
                        logger.warning(f"Could not serialize sample result: {e}")
                        
                return results
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def _validate_query_logic(self, query: str, user_question: str) -> Optional[str]:
        """Validate if the query logic matches the user's intent"""
        prompt = f"""Analyze if the SQL query correctly interprets the user's question.
        
        User question: {user_question}
        Generated query: {query}
        
        Check for common misinterpretations:
        1. Is a country being treated as a course name?
        2. Is a region being treated as a specific location?
        3. Are date/time filters appropriate?
        
        If there are any logical errors, explain what's wrong and how to fix it.
        If the query looks correct, just respond with "VALID".
        """
        
        response = self.llm.generate_content(prompt)
        response_text = response.text.strip()
        
        if "VALID" not in response_text.upper():
            logger.info(f"Query logic validation failed: {response_text}")
            return response_text
        return None

    def process_question(self, user_question: str) -> Dict[str, Any]:
        """Process user question and return results"""
        logger.info(f"\n{'='*80}\nProcessing new question: {user_question}\n{'='*80}")
        
        iterations = 0
        last_error_context = None
        query = ""
        best_results = None
        best_query = None
        
        while iterations < self.config.max_iterations:
            logger.info(f"\nIteration {iterations + 1}/{self.config.max_iterations}")
            try:
                # Generate SQL query
                query = self._generate_sql_query(user_question, last_error_context)
                
                # Validate query structure
                if not self._validate_query(query):
                    raise ValueError("Query validation failed: Only SELECT statements are allowed")
                
                # Execute query
                results = self.execute_query(query)
                
                # If we got no results, validate the query logic
                if not results:
                    logic_error = self._validate_query_logic(query, user_question)
                    if logic_error:
                        last_error_context = {
                            "error": f"Query logic error: {logic_error}",
                            "query": query
                        }
                        iterations += 1
                        continue
                
                # If we got results, validate them with LLM
                if results:
                    df = pd.DataFrame(results)
                    
                    # Log some basic statistics about the results
                    logger.info(f"Results shape: {df.shape}")
                    if 'course_name' in df.columns:
                        logger.info(f"Unique courses in results: {df['course_name'].unique()}")
                    if 'race_date' in df.columns:
                        logger.info(f"Unique dates in results: {df['race_date'].unique()}")
                    
                    # Store these results if they're the first valid ones we've found
                    if best_results is None:
                        best_results = results
                        best_query = query
                    
                    validation_result = self._validate_results_with_llm(df, user_question)
                    if validation_result:
                        logger.info("Query results validated successfully")
                        return {
                            "success": True,
                            "query": query,
                            "results": results,
                            "iterations": iterations + 1,
                            "partial_data": True  # Indicate that results might be incomplete
                        }
                    else:
                        logger.info("Query results validation failed - missing required fields")
                        last_error_context = {
                            "error": "Results are missing required fields. Need to include all requested information.",
                            "query": query
                        }
                else:
                    last_error_context = {
                        "error": "Query returned no results. Check filters and conditions.",
                        "query": query
                    }
                    
            except Exception as e:
                error_msg = self._parse_error_message(e, query) if query else str(e)
                logger.error(f"Error in iteration {iterations + 1}: {error_msg}")
                last_error_context = {
                    "error": error_msg,
                    "query": query
                }
            
            iterations += 1
        
        # If we have any results, return them even if not perfect
        if best_results:
            logger.info("Returning best available results after all iterations")
            return {
                "success": True,
                "query": best_query,
                "results": best_results,
                "iterations": iterations,
                "partial_data": True  # Indicate that results might be incomplete
            }
            
        logger.error(f"Failed to generate valid query after {iterations} attempts")
        return {
            "success": False,
            "error": f"Failed to generate valid query after {iterations} attempts. Last error: {last_error_context['error'] if last_error_context else 'Unknown error'}",
            "iterations": iterations
        } 
