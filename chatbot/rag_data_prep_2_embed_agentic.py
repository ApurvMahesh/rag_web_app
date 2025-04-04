
from sentence_transformers import SentenceTransformer
from transformers import pipeline


from transformers import pipeline
import cohere
from rag_data_prep_1_json import query_sales_db  # Import your SQL-based query function

# Load the generator model (e.g., OpenAI's GPT or HuggingFace)
generator = pipeline("text2text-generation", model="t5-large")

# Initialize Cohere
co = cohere.Client('2QqVvvlG7rs36B51L96YXShsV5hMYw2BV8QS1rqi')


# **New retrieval function using SQL instead of FAISS**
def retrieve_sales_data(query):
    results = query_sales_db(query)  # Query the SQL database
    print("Retrieved Sales Data:", results)
    return results


# **RAG Pipeline using SQL retrieval**
def rag_pipeline(query, generator_model):
    retrieved_data = retrieve_sales_data(query)
    
    # Format retrieved documents into context
    context = "\n".join([
        f"- {row['Region']} region sold {row['Units Sold']} units of {row['Brand']} {row['Product Category']} on {row['Date']}."
        for row in retrieved_data
    ])

    return context


# **Cohere AI Agent using SQL data retrieval**
def cohere_agent(query):
    system_prompt = (
        "You are an AI sales assistant with access to real-time e-commerce data. "
        "Your job is to analyze sales trends, identify patterns, and recommend strategies. "
        "Use the retrieved data to generate actionable insights."
    )

    retrieved_data = retrieve_sales_data(query)

    cohere_response = co.generate(
        model="command-r-plus-08-2024",
        prompt=f"{system_prompt}\n\nQuery: {query}\nSales Data:\n{retrieved_data}\n\nProvide recommendations:",
        max_tokens=1000,
    )

    return cohere_response.generations[0].text


# **Run Multi-Agent Workflow with SQL Retrieval**
def run_multi_agent(query):
    print("Running multi-agent workflow...")
    retrieved = retrieve_sales_data(query)
    analysis = co.generate(
        model="command-r-plus-08-2024",
        prompt=f"Analyze this data and extract key sales trends:\n{retrieved}",
        max_tokens=1000,
    ).generations[0].text

    strategy = co.generate(
        model="command-r-plus-08-2024",
        prompt=f"Based on this analysis, recommend a sales strategy:\n{analysis}",
        max_tokens=1000,
    ).generations[0].text

    return strategy


# **Test Example Query**
# query = """
# SELECT * FROM sales 
# WHERE "Product Category" = 'Electronics' 
# AND Region = 'North'
# """
# # query = "What marketing strategies should I use for Electronics in the North region?"
# response = cohere_agent(query)
# print("Generated Response:", response)










# def retrieve_sales_data(query):
#     response = rag_pipeline(query, SentenceTransformer('all-MiniLM-L6-v2'), index, documents, generator)
#     print("retrieved sales data")
#     return response

# def cohere_agent(query):
#     system_prompt = (
#         "You are an AI sales assistant with access to real-time e-commerce data. "
#         "Your job is to analyze sales trends, identify patterns, and recommend strategies. "
#         "Use the retrieved data to generate actionable insights."
#     )

#     retrieved_data = retrieve_sales_data(query)
    
#     cohere_response = co.generate(
#         model="command-r-plus-08-2024",
#         prompt=f"{system_prompt}\n\nQuery: {query}\nSales Data:\n{retrieved_data}\n\nProvide recommendations:",
#         max_tokens=1000,
#     )
    
#     return cohere_response.generations[0].text

import pickle

class Memory:
    def __init__(self):
        self.history = []

    def add(self, query, response):
        self.history.append({"query": query, "response": response})
        with open("memory.pkl", "wb") as f:
            pickle.dump(self.history, f)

    def get_memory(self):
        return self.history

memory = Memory()


def cohere_agent_with_memory(query):
    past_conversations = "\n".join([f"Q: {m['query']}\nA: {m['response']}" for m in memory.get_memory()])

    system_prompt = (
        "You are an AI sales assistant. Use past conversations and sales data to answer queries effectively."
    )

    retrieved_data = retrieve_sales_data(query)

    full_prompt = f"{system_prompt}\n\nPrevious Conversations:\n{past_conversations}\n\nQuery: {query}\nSales Data:\n{retrieved_data}\n\nProvide recommendations:"

    cohere_response = cohere_agent(full_prompt)

    memory.add(query, cohere_response)

    self_evaluate(cohere_response)
    
    return cohere_response


# Self-Reflection

def take_autonomous_action(query):
    insight = cohere_agent_with_memory(query)
    
    if "low sales" in insight:
        return "Discount applied to improve sales."
    
    return "No action needed."



def self_evaluate(response):
    critique_prompt = f"Evaluate this response on a scale of 1-10, where 10 is perfect and 1 is very poor:\n{response}\nScore:"

    critique = co.generate(
        model="command-r-plus-08-2024",
        prompt=f"Query: {critique_prompt}",
        max_tokens=1000,
    )
    print("Critique: ", critique)
    score = int(critique.generations[0].text.strip())

    return score

def feedback_loop(query, max_iterations=5, min_score=8):
    iteration = 0
    while iteration < max_iterations:
        response = cohere_agent(query)
        score = self_evaluate(response)

        print(f"Iteration {iteration+1}: Score {score}")
        
        if score >= min_score:
            return response  # Stop when quality is high enough

        iteration += 1

    return response  # Return last response if max iterations reached







def cohere_retriever(query):
    return retrieve_sales_data(query)

def cohere_analyzer(data):
    return co.generate(
        model="command-r-plus-08-2024",
        prompt=f"Analyze this data and extract key sales trends:\n{data}",
        max_tokens=1000,
    ).generations[0].text

def cohere_strategist(analysis):
    return co.generate(
        model="command-r-plus-08-2024",
        prompt=f"Based on this analysis, recommend a sales strategy:\n{analysis}",
        max_tokens=1000,
    ).generations[0].text

def run_multi_agent(query):
    retrieved = cohere_retriever(query)
    analysis = cohere_analyzer(retrieved)
    strategy = cohere_strategist(analysis)
    print("ran multi agent")
    return strategy