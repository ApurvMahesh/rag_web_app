
from sentence_transformers import SentenceTransformer
import faiss
import json
from transformers import pipeline
import torch


# Load embeddings, documents, and FAISS index
embeddings = torch.load("sales_data_embeddings.pt")
index = faiss.read_index("sales_data_index.faiss")
with open("sales_data_documents.json", "r") as f:
    documents = json.load(f)

# Load the generator (e.g., OpenAI's GPT or HuggingFace)
generator = pipeline("text2text-generation", model="t5-large")

# Define the retrieval function
def retrieve_documents(query, model, index, documents, top_k=5):
    # Generate query embedding
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    results = [documents[idx] for idx in indices[0]]
    return results


# Build RAG Model
####################
import cohere

co = cohere.Client('2QqVvvlG7rs36B51L96YXShsV5hMYw2BV8QS1rqi')

def rag_pipeline(query, retriever_model, index, documents, generator_model, top_k=5):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, retriever_model, index, documents, top_k)
    
    # Debug retrieved documents
    print("Retrieved Documents:")
    for doc in retrieved_docs:
        print(doc)
    
    # Format retrieved documents into context
    context = "\n".join(
        [f"- {doc['Region']} region sold {doc['Units Sold']} units of {doc['Brand']} {doc['Product Category']} on {doc['Date']}." for doc in retrieved_docs]
    )
    
    # # Create generator input
    # input_text = (
    #     f"Query: {query}\n"
    #     f"Relevant Context:\n{context}\n\n"
    #     "Provide actionable insights based on the query and context.\n\n"
    # )
    
    # # Debug input to generator
    # print("Final Input to Generator:")
    # print(input_text)
    
    # # Generate response
    # # response = generator_model(input_text, max_length=1000, do_sample=True)[0]['generated_text']
    # response = co.generate(
    #     model="command-r-plus-08-2024",
    #     prompt= input_text,
    #     max_tokens=50
    # );
    return context

# Example query
query = "What marketing strategies should I use for Electronics in the North region?"

def getResponseForTheQueryText(query):
    response = rag_pipeline(query, SentenceTransformer('all-MiniLM-L6-v2'), index, documents, generator)
    print("Generated Context : ", response)
    return response


# response = rag_pipeline(query, SentenceTransformer('all-MiniLM-L6-v2'), index, documents, generator)
# print("Generated Response : ", response)









def retrieve_sales_data(query):
    response = rag_pipeline(query, SentenceTransformer('all-MiniLM-L6-v2'), index, documents, generator)
    print("retrieved sales data")
    return response

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
        max_tokens=150,
    )
    
    return cohere_response.generations[0].text

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

    response_text = cohere_response.generations[0].text
    memory.add(query, response_text)

    self_evaluate(response_text)
    
    return response_text


# Self-Reflection

def take_autonomous_action(query):
    insight = cohere_agent_with_memory(query)
    
    if "low sales" in insight:
        return "Discount applied to improve sales."
    
    return "No action needed."



def self_evaluate(response):
    critique_prompt = f"Evaluate this response on a scale of 1-10, where 10 is perfect and 1 is very poor:\n{response}\nScore:"

    critique = cohere_agent_with_memory(critique_prompt)

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
        max_tokens=150,
    ).generations[0].text

def cohere_strategist(analysis):
    return co.generate(
        model="command-r-plus-08-2024",
        prompt=f"Based on this analysis, recommend a sales strategy:\n{analysis}",
        max_tokens=150,
    ).generations[0].text

def run_multi_agent(query):
    retrieved = cohere_retriever(query)
    analysis = cohere_analyzer(retrieved)
    strategy = cohere_strategist(analysis)
    print("ran multi agent")
    return strategy