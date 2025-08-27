# Conversational BI Agent & Sales Forecasting API

This project provides a powerful, AI-driven tool for sales analytics, allowing non-technical users to query sales data using natural language and generate on-demand time-series forecasts.

## Overview

The goal of this project is to bridge the gap between complex sales data and actionable business insights. Instead of relying on manual SQL queries or static dashboards, this tool empowers users to ask questions like *"What were the total sales for Office Supplies last month?"* or *"Show me the top 5 selling products"* and get immediate answers. Furthermore, it leverages Facebook's Prophet model to deliver robust sales forecasts through a simple API.

This repository demonstrates skills in full-stack development, machine learning model deployment (time-series), and building modern AI applications with Large Language Models (LLMs).

## Key Features

* **Conversational Data Analysis**: Utilizes a LangChain agent to translate natural language questions into executable pandas queries on sales data.
* **Predictive Sales Forecasting**: Implements a Prophet time-series model to predict future sales for any given product category.
* **RESTful API**: Built with Flask, providing structured endpoints for forecasting, data visualization, and agent-based queries.
* **Dynamic Plot Generation**: Generates and serves forecast visualizations on-the-fly.

## Technology Stack

* **Backend**: Python, Flask
* **AI & Machine Learning**: LangChain, OpenAI, Prophet, Scikit-learn, Pandas
* **Data Visualization**: Matplotlib
* **Environment**: Git, Virtual Environments

## Setup and Installation

Follow these steps to get the project running on your local machine.

**1. Clone the Repository**
```bash
git clone [https://github.com/Theja965/Conversational-BI-Agent.git](https://github.com/Theja965/Conversational-BI-Agent.git)
cd Conversational-BI-Agent
```

**2. Create and Activate a Virtual Environment**
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**
All required libraries are listed in the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Set Environment Variable**
This project requires an OpenAI API key. Set it as an environment variable for security.

```bash
# For macOS/Linux
export OPENAI_API_KEY="your_openai_api_key_here"

# For Windows
set OPENAI_API_KEY="your_openai_api_key_here"
```
**Important:** Do not hardcode your API key in the source code.

## Usage

**1. Run the Flask Application**
```bash
python main.py
```
The server will start, typically on `http://127.0.0.1:5001`.

**2. Interact with the API**
You can use the provided `test_api.py` script or a tool like Postman to interact with the endpoints.

To run the test script:
```bash
python test_api.py
```
This script demonstrates how to call the `/forecast` endpoint for the "Office Supplies" category.

### API Endpoints

* **`POST /forecast`**: Returns a JSON object with sales forecast data.
    * **Body**: `{ "category": "Office Supplies", "periods": 30 }`
* **`POST /plot`**: Returns a PNG image of the sales forecast plot.
    * **Body**: `{ "category": "Furniture", "periods": 90 }`
* **`POST /agent`**: Allows you to ask natural language questions about the sales data.
    * **Body**: `{ "query": "What is the total sales for the Furniture category?" }`
