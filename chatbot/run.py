import os

# Running the first Python file
# os.system("python rag_data_gen.py")

# Running the second Python file
os.system("python rag_data_prep_1_json.py")

# Now running the Flask app (app.py)
os.system("python app.py")