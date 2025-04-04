from flask import Flask, render_template, request, session
import cohere  # Replace with OpenAI or other APIs if needed
import rag_data_prep_2_embed_agentic
import rag_data_prep_1_json
import re

app = Flask(__name__)

# Initialize the Cohere client with your API key
co = cohere.Client('2QqVvvlG7rs36B51L96YXShsV5hMYw2BV8QS1rqi')


# Create an empty list to store chat history
app.secret_key = '5t7678634g798798374'  # Make sure to change this

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

# For Query Matching
def needs_context(query):
    # Define which queries need context based on keywords or patterns
    # Here, we're checking if the query involves specific business tasks like sales, inventory, etc.
    context_required_queries = [
        r"(sales|revenue|performance|inventory|stock|staffing|product)",  # Pattern for sales and inventory-related queries
        r"(holiday|weather|event|special)",  # Pattern for holiday/event-related queries
    ]
    
    for pattern in context_required_queries:
        if re.search(pattern, query, re.IGNORECASE):  # Ignore case when matching
            return True  # Return True if the query matches any pattern
    
    return False  # Return False if no match found


@app.route('/ask', methods=['POST'])
def ask():
    if 'chat_history' not in session:
        session['chat_history'] = []

    user_input = request.form['user_input']
    schema = rag_data_prep_1_json.query_db_info()
    print("Schema:", schema)
    promptFirst = f"Query: The User Input Query Is {user_input}, reformat the user query to keep it short and sensible. Context: {session['chat_history']}, TableName: sales, sales: {schema} is just to show the data column names and the type of data stored"
    firstResponse = co.generate(
            model='command-r-plus-08-2024',  # Specify your desired model
            prompt=promptFirst,
            max_tokens=1000
        )
    firstResponse = firstResponse.generations[0].text.strip()
    print("firstResponse:", firstResponse)
    prompt = f"Query: The User Input Query Is {firstResponse}, if the query needs data from db, return only the sql query in the output based on table schema & data type given or if the sql query can not be generated then return NO. Try your best to generate SQL query rather than saying NO, Context: {session['chat_history']}, TableName: sales, sales: {schema} is just to show the data column names and the type of data stored"
    midResponse = co.generate(
            model='command-r-plus-08-2024',  # Specify your desired model
            prompt=prompt,
            max_tokens=1000
        )
    midResponse = midResponse.generations[0].text.strip()
    print("Mid Response:", midResponse)
    if(midResponse == "NO"):    
        prompt = f"Query: The User Input Query Is {user_input}, Context: {session['chat_history']}"
    else:
        midResponse = midResponse.replace("```sql", "").strip()
        midResponse = midResponse.replace("```", "").strip()
        print(midResponse)
        data = rag_data_prep_1_json.query_sales_db(midResponse)
        print("Data:", data)
        prompt = f"Query: The User Input Query Is {user_input}, Context: {session['chat_history']}, Data: {data}"
    response = co.generate(
            model='command-r-plus-08-2024',  # Specify your desired model
            prompt=prompt,
            max_tokens=1000
        )
    response = response.generations[0].text.strip()
    chatbot_response = response
    session['chat_history'].append({"user": user_input, "bot": chatbot_response})
    session.modified = True  # Mark session as modified so that Flask saves it

    return render_template('index.html', chat_history=session['chat_history'])

if __name__ == '__main__':
    app.run(debug=True)
