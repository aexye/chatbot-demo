import streamlit as st
import pandas as pd
from sql_agent import SQLAgent, SQLAgentConfig
from datetime import datetime
import json

def analyze_data_with_llm(df: pd.DataFrame, question: str, model) -> str:
    """Generate natural language analysis of the query results using LLM"""
    # Convert dataframe info to string
    data_info = df.to_json(orient='records', date_format='iso')
    
    system_prompt = """You are analyzing horse racing data. Provide a brief, focused summary of the key points.
    
    Guidelines:
    1. Keep it to 2-3 short paragraphs maximum
    2. Focus on the most interesting or unusual aspects of the data
    3. If there are any standout numbers or statistics, mention them
    4. If relevant, note any potential betting angles
    5. Use plain, direct language
    
    Remember: Be concise and highlight only the most noteworthy findings."""
    
    prompt = f"""Question asked: {question}

Data:
{data_info}

Provide a quick analysis focusing on the most interesting aspects."""

    response = model.generate_content(system_prompt + "\n\n" + prompt)
    return response.text

def main():
    # Initialize configuration
    config = SQLAgentConfig(
        project_id="data-gaming-425312",
        dataset_id="horse_chatbot",
        credentials_dict=st.secrets["gcp_service_account"],
        model_name="gemini-1.5-flash-002",
        temperature=0,
        max_iterations=10
    )
    
    # Create SQL agent
    agent = SQLAgent(config)
    
    # Example schema for the racecards table
    racecards_schema = """
    Table: data-gaming-425312.horse_chatbot.racecards
    
    Columns:
    - Race details: race_id, race_name, race_date, race_type, race_class, race_distance, course_name
    - Horse details: horse_name, horse_age, horse_sex, horse_draw, rating
    - Performance: runner_odds, avg_form, last_position, days_from_last_race
    - Personnel: jockey_name
    - Win rates: hr_win_rate, jc_win_rate, tr_win_rate (horse, jockey, trainer)
    - Recent performance: jc_14d_win_rate, tr_14d_win_rate
    
    Notes:
    - one row = one horse per one race
    - All rates are decimal values between 0 and 1
    """
    
    # Streamlit UI
    st.title("Horse Racing SQL Assistant")
    
    # User input
    user_question = st.text_input("Ask a question about horse racing:", 
                                "Show me all races at Ascot today with their times and number of runners")
    
    if st.button("Get Answer"):
        with st.spinner("Processing your question..."):
            # Process the question
            result = agent.process_question(user_question, racecards_schema)
            
            # Display results
            if result["success"]:
                st.success(f"Found answer in {result['iterations']} iterations")
                
                # Show the SQL query in an expandable section
                with st.expander("View SQL Query"):
                    st.code(result["query"], language="sql")
                
                # Convert results to DataFrame for analysis
                df = pd.DataFrame(result["results"])
                
                # Display raw data in a table
                st.write("Raw Data:")
                st.dataframe(df)
                
                # Generate and display LLM analysis
                with st.expander("View Analysis", expanded=True):
                    analysis = analyze_data_with_llm(df, user_question, agent.llm)
                    st.write(analysis)
                
            else:
                st.error(result["error"])
                st.write(f"Attempted {result['iterations']} times")

if __name__ == "__main__":
    main() 
