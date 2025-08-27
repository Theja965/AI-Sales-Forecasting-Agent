import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import openai
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    # Make sure your CSV file is named exactly this or change the name here
    full_df = pd.read_csv("Cleaned_sales_data.csv") 
except FileNotFoundError:
    full_df = None
    print("ERROR: Cleaned_sales_data.csv not found!")

# --- THE ONLY TOOL ---
@tool
def get_sales_forecast():
    """
    Use this tool ONLY when the user asks for 'the forecast', 'a prediction', or a 'projection'.
    This tool takes no arguments.
    """
    if full_df is None:
        return "Error: Sales data is not loaded."
    
    # Simplified data cleaning logic
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

# --- THE MINIMAL AGENT ---
tools = [get_sales_forecast]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- FLASK APP ---
app = Flask(__name__)
CORS(app)
@app.route("/api/chat", methods=["POST"])
def chat():
    user_question = request.json.get("question")
    response = agent_executor.invoke({"input": user_question})
    return jsonify({"answer": response['output']})

if __name__ == "__main__":
    app.run(debug=True, port=5000)