"""
# 🤖 AI-Powered Sales Analyst Agent

### Overview
This application creates a conversational AI agent designed to act as a Sales Data Analyst. It integrates OpenAI’s GPT-3.5, LangChain, and FB Prophet to allow users to ask natural language questions about sales data.

### How it Works:
* **Natural Language to Action:** The agent determines if it needs to run a Prophet forecast or a manual projection based on the user's query.
* **Tool Use:** It uses a dedicated `get_prophet_forecast` tool to interact with Python dataframes.
* **Flask API:** The logic is wrapped in a Flask web server, making it ready for integration into a dashboard or chat interface.

### Tech Stack:
Python, Flask, LangChain, OpenAI API, Pandas, Prophet.
"""


import os
import pandas as pd
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from dotenv import load_dotenv
from prophet import Prophet
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    full_df = pd.read_csv("Cleaned_sales_data.csv")
except FileNotFoundError:
    print("CRITICAL ERROR: product_sales.csv not found! The app cannot run.")
    full_df= None

# Forecasting Tools 

#AI_Driven Prophet Forecast
@tool
def get_prophet_forecast():
    """
    Runs a full, data-driven Prophet forecast model on the historical sales data 
    and returns a 6-month sales forecast. Use this tool ONLY when the user asks 
    for 'the forecast', 'a prediction', or 'a projection' and does NOT 
    mention a specific growth rate.
    """
    if full_df is None: return "Error: Sales data not loaded."
    
    df_copy = full_df.copy()
    df_copy['AcceptedDate'] = pd.to_datetime(df_copy['AcceptedDate'], errors='coerce')
    df_copy.dropna(subset=['AcceptedDate'], inplace=True)
    df_copy.set_index('AcceptedDate', inplace=True)
    df_copy.sort_index(inplace=True)
    monthly_sales = df_copy['NetTotal'].resample('M').sum()
    modeling_data = monthly_sales.asfreq('M').fillna(0)
    
    prophet_df = modeling_data.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    model_prophet = Prophet()
    model_prophet.fit(prophet_df)
    
    future_dates = model_prophet.make_future_dataframe(periods=6, freq='M')
    forecast = model_prophet.predict(future_dates)
    
    return forecast[['ds', 'yhat']].tail(6).to_dict('records')

#User_driven Manual Projection 
@tool
def get_manual_projection(growth_rate: float):
    """
    Calculates a simple 6-month projection based on a user-provided growth rate percentage.
    Use this tool for 'what-if' scenarios or when the user provides a specific 
    growth rate to test.
    """
    if full_df is None:
        return "Error: Sales data is not loaded."
    df_copy = full_df.copy()
    df_copy['AcceptedDate'] = pd.to_datetime(df_copy['AcceptedDate'], errors='coerce')
    df_copy.dropna(subset=['AcceptedDate'], inplace=True)
    df_copy.set_index('AcceptedDate', inplace=True)
    df_copy.sort_index(inplace=True)
    monthly_sales = df_copy['NetTotal'].resample('M').sum()
    modeling_data = monthly_sales.asfreq('M').fillna(0)

    last_known_sales = modeling_data.iloc[-1]
    projections = []
    current_sales = last_known_sales
    current_date = modeling_data.index[-1]

    for _ in range(6):
        current_sales *= (1 + growth_rate / 100)
        current_date = current_date + pd.DateOffset(months=1)
        projections.append({'ds': current_date.strftime('%Y-%m-%d'), 'yhat': current_sales})
    return projections


# The AI Agent 
tools = [get_prophet_forecast, get_manual_projection]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and precise sales data analyst."), 
    ("user", "{input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad"), 
])
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#Flask Application and API Setup
app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    if full_df is None:
        return jsonify({"error": "Server Error: Sales data is not loaded"}), 500

    try:
        # Instead of calling OpenAI directly, we call our agent executor
        response = agent_executor.invoke({
            "input": user_question
        })
        
        # THE FIX: Extract just the 'output' string from the response dictionary
        ai_response = response['output']

    except Exception as e:
        print(f"Error during agent execution: {e}")
        ai_response = f"An error occurred: {e}"

    return jsonify({"answer": ai_response})

#Run the application
if __name__ == "__main__":
    app.run(debug=True, port=5000)

