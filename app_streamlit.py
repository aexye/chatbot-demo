import os
from openai import OpenAI
import instructor
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from dotenv import load_dotenv
import streamlit as st
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from datetime import datetime
from pydantic import Field
import warnings
from langchain.prompts import PromptTemplate
from sqlalchemy.dialects import registry
from langgraph.prebuilt import create_react_agent
from google.oauth2 import service_account


prompt_template = """
You are an agent designed to interact with a SQL database.

Given an input question, create a syntactically correct BigQuery query to run, then look at the results of the query and return the answer.
Don't use "```" at the beginning or end of your query. In BigQuery always construct FROM with schema.table.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.

Today's date is {today}.

You have access to the following tables (both tables have granularity of horse in race = one row per horse per race):
data-gaming-425312.horse_chatbot.results - contains results for all races - use for historical performance data
data-gaming-425312.horse_chatbot.racecards - contains racecards for today's races - {today}
course_name - city of the race
"""

# Schema definitions
class SQLToolInputSchema(BaseAgentInputSchema):
    """Input schema for the SQL Tool."""
    chat_message: str = Field(..., description="User's question about horse racing")
    is_sql_query: bool = Field(False, description="If question is about historical data, you can use the SQL tool to get the data")

class SQLToolOutputSchema(BaseAgentOutputSchema):
    """Output schema for the SQL Tool."""
    sql_result: str = Field(..., description="The result from the SQL query")
    chat_message: str = Field(default="", description="Chat message from the SQL tool")

class HorseRacingAgentInputSchema(BaseAgentInputSchema):
    """Input schema for the Horse Racing Agent."""
    chat_message: str = Field(..., description="User's question about horse racing")

class HorseRacingAgentOutputSchema(BaseAgentOutputSchema):
    """Output schema for the Horse Racing Agent."""
    chat_message: str = Field(..., description="User's question about horse racing")
    is_sql_query: bool = Field(False, description="If question is about historical data, you can use the SQL tool to get the data")

class AnswerAgentOutputSchema(BaseAgentOutputSchema):
    """Output schema for the Answer Agent."""
    chat_message: str = Field(..., description="Answer to the user's question")

# Initialize components
memory = AgentMemory()
today = datetime.now().strftime("%Y-%m-%d")

system_prompt_generator = SystemPromptGenerator(
    background=[
        "Your name is Henry, a horse racing expert.",
        "You are a knowledgeable expert on horse racing, equestrian sports, and the horse racing industry.",
    ],
    steps=[
        "Your role is to rephrase the user's question into a question that can be answered by the SQL tool.",
        "If the user asks a question about horse racing data or about horse/jockey/trainer performance, you can use the SQL tool to get the data.",
        "If the user's question is about historical data, you should use the SQL tool to get the data.",
        "If the user's question is not about historical data, just pass the question to the answer agent."
    ],
    output_instructions=[
        "If the user's question is about historical data, you should use the SQL tool to get the data.",
        "If the user's question is not about historical data, just pass the question to the answer agent."
    ]
)

system_prompt_generator_answer = SystemPromptGenerator(
    background=[
        "Your name is Henry, a horse racing expert.",
        "You are a knowledgeable expert on horse racing, equestrian sports, and the horse racing industry.",
        "You have extensive knowledge of racing history, famous horses, jockeys, trainers, and racing strategies.",
        "You can provide insights on betting, handicapping, race types, track conditions, and horse care.",
        "Your knowledge is based on UK horse racing. You have only access to UK horse racing data.",
    ],
    steps=[
        "If the user asks a question that is not about horse racing, politely decline to answer and ask if you can help with something else.",
        "Analyze the user's question about horse racing or related topics.",
        "Draw upon expert knowledge to provide accurate, detailed information.",
        "Explain concepts clearly while considering the user's level of familiarity with racing.",
        "If relevant, provide a follow-up question to the user.",
    ],
    output_instructions=[
        "Provide clear, factual information about horse racing topics.",
        "Use proper racing terminology while explaining concepts accessibly, but use in words that are easy to understand.",
        "Be precise about racing rules, betting concepts, and safety considerations.",
    ]
)

def sql_tool(input_schema: SQLToolInputSchema) -> SQLToolOutputSchema:
    project_id = "data-gaming-425312"
    dataset_id = "horse_chatbot"
    connection_string = f"bigquery://{project_id}/{dataset_id}"
    
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    engine = create_engine(
        connection_string,
        credentials=credentials
    )
    
    db = SQLDatabase(engine)
    prompt = prompt_template.format(dialect="BigQuery", top_k=20, today=today)
    llm = ChatVertexAI(
        model_name="gemini-1.5-flash-002", 
        temperature=0, 
        api_key=st.secrets["gemini_api_key"],
        convert_system_message_to_human=True
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_agent = create_react_agent(
        llm,
        tools=toolkit.get_tools(),
        state_modifier=prompt,
    )

    try:
        response = sql_agent.invoke({
            "messages": [{"role": "user", "content": input_schema.chat_message}]
        })
        
        # Extract the actual content from the LangChain response
        if isinstance(response, dict):
            result = response.get("output", "")
            if not result:
                messages = response.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict):
                        content = last_message.get("content", "")
                        result = content.split(" additional_kwargs=")[0].strip()
                    else:
                        result = str(last_message)
        else:
            result = str(response)
            
        result = result.replace("content=", "").strip('"')
        print(f"Cleaned result: {result}")  # For debugging
        
        return SQLToolOutputSchema(
            sql_result=result,
            chat_message=result
        )
    except Exception as e:
        print(f"Error in SQL tool: {str(e)}")
        error_message = "Sorry, I encountered an error while querying the database. Please try rephrasing your question."
        return SQLToolOutputSchema(
            sql_result=error_message,
            chat_message=error_message
        )

# Add before the main() function
# Initialize the main agent with schema alignment
api_key = os.getenv("GEMINI_API_KEY")
client = instructor.from_openai(
    OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
    mode=instructor.Mode.JSON,
)
model = "gemini-1.5-flash-latest"

agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model=model,
        memory=memory,
        max_tokens=2048,
        system_prompt_generator=system_prompt_generator,
        input_schema=HorseRacingAgentInputSchema,
        output_schema=HorseRacingAgentOutputSchema
    )
)

answer_agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model=model,
        memory=memory,
        max_tokens=2048,
        system_prompt_generator=system_prompt_generator_answer,
        input_schema=HorseRacingAgentInputSchema,
        output_schema=AnswerAgentOutputSchema
    )
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.memory = AgentMemory()
        # Add initial message
        initial_message = HorseRacingAgentOutputSchema(chat_message="Hello! How can I assist you today?")
        st.session_state.memory.add_message("assistant", initial_message)
        st.session_state.messages.append({"role": "assistant", "content": initial_message.chat_message})

def main():
    st.title("Horse Racing Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about horse racing?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Process the user's input through the agent
        input_schema = HorseRacingAgentInputSchema(chat_message=prompt)
        response = agent.run(input_schema)
        
        final_message = ""
        
        # If the agent's response includes a SQL query, execute it using the SQL tool
        if response.is_sql_query:
            with st.spinner('Querying database...'):
                sql_result = sql_tool(SQLToolInputSchema(
                    chat_message=response.chat_message,
                    is_sql_query=response.is_sql_query
                ))
                
                # Clean up the response
                result = sql_result.sql_result
                if isinstance(result, str):
                    # Remove metadata and format the message
                    clean_result = result.split(" additional_kwargs=")[0]
                    clean_result = clean_result.strip('"\' ')
                    clean_result = clean_result.replace("content=", "")
                    final_message = clean_result
                else:
                    final_message = str(result)
                
                st.session_state.memory.add_message("assistant", {"SQLToolOutputSchema": sql_result})
        else:
            final_message = answer_agent.run(input_schema)
            final_message = final_message.chat_message
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_message})
        with st.chat_message("assistant"):
            st.write(final_message)

if __name__ == "__main__":
    # Load environment variables and initialize components
    load_dotenv()
    
    # Initialize all your components
    registry.register('bigquery', 'sqlalchemy_bigquery', 'BigQueryDialect')
    warnings.filterwarnings("ignore", message="Cannot create BigQuery Storage client")
    
    # Run the Streamlit app
    main() 