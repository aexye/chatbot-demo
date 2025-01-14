from typing import Optional, Dict, Any, List, Set
import logging
from dataclasses import dataclass
import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect
import streamlit as st
from datetime import datetime
import re
import json
from google.oauth2 import service_account
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
    credentials_dict: Dict[str, Any]
    model_name: str = "gemini-1.5-flash-002"
    temperature: float = 0
    max_iterations: int = 3
    location: str = "us-central1"

class SQLAgent:
    def __init__(self, config: SQLAgentConfig):
        self.config = config
        # Create credentials first since it's needed by other methods
        self.credentials = self._create_credentials()
        # Then create engine and other components that might need credentials
        self.engine = self._create_engine()
        self.db = SQLDatabase(self.engine)
        self.llm = self._create_llm()
        self.tried_queries: Set[str] = set()
        self.previous_attempts: List[Dict[str, Any]] = []
        self.table_columns = self._get_table_columns()
        logger.info(f"Available columns: {self.table_columns}")
        
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
    
    def _get_table_columns(self) -> Dict[str, Dict[str, Any]]:
        """Get all columns and their descriptions from BigQuery schema"""
        try:
            from google.cloud import bigquery
            
            # Create BigQuery client
            client = bigquery.Client(
                project=self.config.project_id,
                credentials=self.credentials
            )
            
            table_info = {}
            # Get full table path
            table_path = f"{self.config.project_id}.{self.config.dataset_id}"
            
            # List all tables in the dataset
            tables = client.list_tables(table_path)
            
            for table in tables:
                # Get the table object to access schema
                table_obj = client.get_table(table.reference)
                table_info[table.table_id] = {
                    'columns': [],
                    'descriptions': {}
                }
                
                # Extract column names and descriptions from schema
                for field in table_obj.schema:
                    table_info[table.table_id]['columns'].append(field.name)
                    if field.description:
                        table_info[table.table_id]['descriptions'][field.name] = field.description
                    
                logger.info(f"Found table {table.table_id} with columns: {table_info[table.table_id]['columns']}")
                
            return table_info
            
        except Exception as e:
            logger.error(f"Error fetching schema from BigQuery: {str(e)}")
            return {}

    def _push_schema_info(self, schema: str) -> str:
        """Add detailed column descriptions to the schema on first iteration"""
        enhanced_schema = schema + "\n\nDetailed column descriptions:\n"
        
        for table, info in self.table_columns.items():
            enhanced_schema += f"\nTable: {table}\n"
            for col in info['columns']:
                description = info['descriptions'].get(col, 'No description available')
                enhanced_schema += f"- {col}: {description}\n"
        
        return enhanced_schema

    def _validate_columns_in_query(self, query: str) -> List[str]:
        """Check if all columns in the query exist in the schema"""
        invalid_columns = []
        
        # Extract all column references from the query
        # This is a simple regex that looks for table alias followed by column name
        column_pattern = r't1\.(\w+)'
        referenced_columns = re.findall(column_pattern, query, re.IGNORECASE)
        
        # Get all available columns from all tables
        all_columns = set()
        for table_info in self.table_columns.values():
            all_columns.update(table_info['columns'])
            
        # Check each referenced column against the schema
        for col in referenced_columns:
            if col not in all_columns:
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
            available_columns = self.table_columns.get('racecards', [])
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

    def _generate_sql_query(self, user_question: str, schema: str, error_context: Optional[Dict] = None) -> str:
        """Generate SQL query using LLM"""
        logger.info(f"Generating SQL query for question: {user_question}")
        if error_context:
            logger.info(f"Previous error context: {json.dumps(error_context, indent=2)}")
            
        system_prompt = """You are a SQL expert specializing in BigQuery queries.
        Given the following schema and user question, create a syntactically correct BigQuery SQL query.
        
        Guidelines:
        1. Use LIKE with wildcards for name searches (e.g. LIKE '%horse_name%')
        2. Only query relevant columns that exist in the schema
        3. When filtering dates, treat them as date types
        4. Never use DML statements (INSERT, UPDATE, DELETE, DROP etc.)
        5. Always use fully qualified table names (project.dataset.table)
        6. Don't use "`" in your query
        7. Include DISTINCT when counting to avoid duplicates
        8. Group by all non-aggregated columns
        9. Order results by time for better readability
        10. If a requested column doesn't exist in the schema:
            - Skip that column and continue with available columns
            - Use alternative columns that provide similar information if available
            - For example, if race_time doesn't exist but race_date does, use race_date
        11. When creating aggregate functions, create a new name for the column
        12. Use the correct rate columns based on the entity:
            - For horse performance: use hr_* columns (hr_win_rate, hr_top_3_rate, etc.)
            - For jockey performance: use jc_* columns (jc_win_rate, jc_14d_win_rate, etc.)
            - For trainer performance: use tr_* columns (tr_win_rate, tr_14d_win_rate, etc.)
            - For combined stats: use jctr_* columns
        13. When the question is about a horse's performance, always prefer hr_* columns over other rates
        """
        
        if error_context:
            system_prompt += "\n\nPrevious attempt failed:"
            system_prompt += f"\nError: {error_context['error']}"
            system_prompt += f"\nFailed query:\n{error_context['query']}"
            if "Invalid columns" in error_context['error']:
                system_prompt += "\nPlease exclude these columns and use only available ones from the schema."
            else:
                system_prompt += "\nPlease fix the issues and try a different approach."
            
        # Add previous attempts to help avoid repetition
        if self.previous_attempts:
            system_prompt += "\n\nPrevious attempts that failed:"
            for i, attempt in enumerate(self.previous_attempts[-3:], 1):  # Show last 3 attempts
                system_prompt += f"\n\nAttempt {i}:"
                system_prompt += f"\nQuery: {attempt['query']}"
                system_prompt += f"\nError: {attempt['error']}"
            system_prompt += "\n\nPlease generate a completely different query that avoids these issues."
            system_prompt += "\nDo not repeat any of the previous queries."
            
        # Add available columns to the prompt
        if self.table_columns:
            system_prompt += "\n\nAvailable columns in tables:"
            for table, info in self.table_columns.items():
                system_prompt += f"\n{table}: {', '.join(info['columns'])}"
                
        # Add specific guidance based on error patterns
        if error_context and "Invalid columns" in error_context['error']:
            invalid_cols = re.findall(r"Invalid columns in query: ([^.]+)", error_context['error'])
            if invalid_cols:
                system_prompt += f"\n\nNote: The following columns are not available and should be excluded: {invalid_cols[0]}"
                system_prompt += "\nPlease modify the query to use only available columns from the schema."
                # Get all available columns for suggestions
                all_columns = set()
                for table_info in self.table_columns.values():
                    all_columns.update(table_info['columns'])
                # Suggest alternatives
                for col in invalid_cols[0].split(', '):
                    suggestions = [c for c in all_columns if col.lower() in c.lower()]
                    if suggestions:
                        system_prompt += f"\nPossible alternatives for '{col}': {', '.join(suggestions)}"
                    else:
                        system_prompt += f"\nNo direct alternatives found for '{col}', please exclude it from the query."
        
        prompt = f"{system_prompt}\n\nSchema:\n{schema}\n\nUser question: {user_question}\n\nGenerate only the SQL query without any explanations."
        
        response = self.llm.generate_content(prompt)
        
        # Extract SQL query from response
        query = str(response.text)
        query = query.strip('`').strip()
        if query.startswith('sql'):
            query = query[3:].strip()
        query = query.replace("`", "")
        
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
                
                # Try to log a sample result, but don't fail if we can't
                if results:
                    try:
                        sample_json = json.dumps(results[0], indent=2, cls=DateTimeEncoder)
                        logger.info(f"Sample result: {sample_json}")
                    except Exception as e:
                        logger.warning(f"Could not serialize sample result: {e}")
                        
                return results
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def process_question(self, user_question: str, schema: str) -> Dict[str, Any]:
        """Main method to process user question and return results"""
        logger.info(f"\n{'='*80}\nProcessing new question: {user_question}\n{'='*80}")
        
        iterations = 0
        last_error_context = None
        last_successful_results = None
        self.previous_attempts = []  # Reset previous attempts for new question
        query = ""  # Initialize query variable
        
        while iterations < self.config.max_iterations:
            logger.info(f"\nIteration {iterations + 1}/{self.config.max_iterations}")
            try:
                # On first iteration, enhance schema with column descriptions
                current_schema = self._push_schema_info(schema) if iterations == 0 else schema
                
                # Generate SQL query
                query = self._generate_sql_query(user_question, current_schema, last_error_context)
                
                # Normalize query for comparison
                normalized_query = self._normalize_query(query)
                
                # Check if we've tried this query before
                if normalized_query in self.tried_queries:
                    logger.warning("Generated a duplicate query, skipping...")
                    # If we have successful results from a previous iteration, return them
                    if last_successful_results is not None:
                        logger.info("Using results from previous successful query")
                        return {
                            "success": True,
                            "query": query,
                            "results": last_successful_results,
                            "iterations": iterations + 1
                        }
                        
                    last_error_context = {
                        "error": "Generated a duplicate query. Please try a different approach.",
                        "query": query
                    }
                    self.previous_attempts.append(last_error_context)
                    iterations += 1
                    continue
                
                self.tried_queries.add(normalized_query)
                logger.info("Query is unique, proceeding with validation")
                
                # Validate query structure and columns
                if not self._validate_query(query):
                    raise ValueError("Query validation failed: Only SELECT statements are allowed")
                
                # Execute query
                results = self.execute_query(query)
                
                # If we got results, consider it a success
                if results:
                    logger.info("Query returned results successfully")
                    last_successful_results = results  # Store successful results
                    return {
                        "success": True,
                        "query": query,
                        "results": results,
                        "iterations": iterations + 1
                    }
                else:
                    last_error_context = {
                        "error": "Query returned no results. Try adjusting filters or conditions.",
                        "query": query
                    }
                    self.previous_attempts.append(last_error_context)
                    logger.warning("Query returned no results, will try again")
                    
            except Exception as e:
                error_msg = self._parse_error_message(e, query) if query else str(e)
                logger.error(f"Error in iteration {iterations + 1}: {error_msg}")
                last_error_context = {
                    "error": error_msg,
                    "query": query
                }
                self.previous_attempts.append(last_error_context)
            
            iterations += 1
            
        logger.error(f"Failed to generate valid query after {iterations} attempts")
        return {
            "success": False,
            "error": f"Failed to generate valid query after {iterations} attempts. Last error: {last_error_context['error'] if last_error_context else 'Unknown error'}",
            "iterations": iterations
        } 
