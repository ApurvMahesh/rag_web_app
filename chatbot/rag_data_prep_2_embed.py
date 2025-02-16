
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

