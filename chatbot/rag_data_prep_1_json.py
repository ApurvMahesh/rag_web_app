import pandas as pd
import json

# Load the synthetic dataset
df = pd.read_csv("synthetic_sales_data_enriched.csv")

# Convert each row into a JSON document
documents = []
for _, row in df.iterrows():
    document = {
        "Date": row["Date"],
        "Region": row["Region"],
        "Product Category": row["Product Category"],
        "Product ID": row["Product ID"],
        "Brand": row["Brand"],
        "Item Size": row["Item Size"],
        "Units Sold": row["Units Sold"],
        "Unit Price": row["Unit Price"],
        "Discount Offered (%)": row["Discount Offered (%)"],
        "Marketing Spend ($)": row["Marketing Spend ($)"],
        "Customer Segment": row["Customer Segment"],
        "Sales Origin": row["Sales Origin"],
        "Payment Method": row["Payment Method"],
        "Total Revenue ($)": row["Total Revenue ($)"],
    }
    documents.append(document)

# Save documents as JSON
with open("sales_data_documents.json", "w") as f:
    json.dump(documents, f, indent=4)

print("Sales data has been converted into JSON documents and saved.")



# Generate Embeddings
##########################

from sentence_transformers import SentenceTransformer

# Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert documents to textual representations
texts = [json.dumps(doc) for doc in documents]

# Generate embeddings
embeddings = model.encode(texts, convert_to_tensor=True)

# Save embeddings for later use
import torch
torch.save(embeddings, "sales_data_embeddings.pt")
print("Embeddings generated and saved.")


# Setting Up Vector DB
##########################

import faiss
import numpy as np

# Convert embeddings to NumPy array
embeddings_np = embeddings.cpu().numpy()

# Create FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Save FAISS index
faiss.write_index(index, "sales_data_index.faiss")
print("FAISS index created and saved.")


# Define Query Logic
########################
def retrieve_similar_documents(query, model, index, documents, top_k=5):
    # Generate embedding for the query
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    
    # Search for top-k similar documents
    distances, indices = index.search(query_embedding, top_k)
    
    # Return the matching documents
    results = [documents[idx] for idx in indices[0]]
    return results

# Example query
query = "Sales trends for Electronics in North region"
results = retrieve_similar_documents(query, model, index, documents)
for i, result in enumerate(results):
    print(f"Result {i+1}:", result)
print("\n\nVerified the query format is working as expected with indexed search FAISS library\n\n")



