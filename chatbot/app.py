from flask import Flask, render_template, request, session
import cohere  # Replace with OpenAI or other APIs if needed
import rag_data_prep_2_embed_agentic
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
    user_input = request.form['user_input']

     # Conditionally add context to the prompt based on the query
    if needs_context(user_input):
        # If context is needed, add it to the prompt
        prompt = (
            f"Query: {user_input}\n"
            f"Relevant Context:\n{rag_data_prep_2_embed_agentic.getResponseForTheQueryText(user_input)}\n\n"
            "Provide actionable insights based on the query and context.\n\n"
            )
        max_tokens = 1000
    else:
        # If no context is needed, provide only the query
        prompt = f"Query: {user_input}"
        max_tokens = 100

    # Send user input to Cohere API and get response
    if needs_context(user_input):
        response = rag_data_prep_2_embed_agentic.feedback_loop(prompt)
    else:
        response = co.generate(
            model='command-r-plus-08-2024',  # Specify your desired model
            prompt=prompt,
            max_tokens=max_tokens
        )
        response = response.generations[0].text.strip()

      #     # Get the generated text
      #     chatbot_response = response.generations[0].text.strip()

      #     # Append the user input and chatbot response to chat history
      #     chat_history.append({"user": user_input, "bot": chatbot_response})

      #     return render_template('index.html', chat_history=chat_history)
      # Store chat history in session
    if 'chat_history' not in session:
        session['chat_history'] = []

    chatbot_response = response
    session['chat_history'].append({"user": user_input, "bot": chatbot_response})
    session.modified = True  # Mark session as modified so that Flask saves it

    return render_template('index.html', chat_history=session['chat_history'])

if __name__ == '__main__':
    app.run(debug=True)
