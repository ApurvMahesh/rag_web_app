import pandas as pd
from sqlalchemy import create_engine


# Load the dataset
def load_and_enrich_data():
    df = pd.read_csv("synthetic_sales_data_enriched.csv")

    # Define SQLite database (You can switch to PostgreSQL or MySQL if needed)
    engine = create_engine("sqlite:///sales_data.db")

    # Store data in SQL database
    df.to_sql("sales", con=engine, if_exists="replace", index=False)

    print("Sales data has been successfully stored in the SQL database.")



# Define query function
def query_sales_db(query):
    engine = create_engine("sqlite:///sales_data.db")
    with engine.connect() as connection:
        result = pd.read_sql(query, connection)
    return result

# def query_db_info():
#     engine = create_engine("sqlite:///sales_data.db")

#     with engine.connect() as connection:
#         result = connection.exec_driver_sql(f"PRAGMA table_info(sales)")  # ✅ Corrected
#         columns = result.fetchall()

#     # Format the output as a string
#     schema_info = "\n".join([f"Column Name: {col[1]}, Data Type: {col[2]}" for col in columns])
    
#     return schema_info

def query_db_info():
    engine = create_engine("sqlite:///sales_data.db")

    with engine.connect() as connection:
        # Get column names and types
        result = connection.exec_driver_sql("PRAGMA table_info(sales)")
        columns = result.fetchall()

        # Get sample data (limit 5 rows)
        data_result = connection.exec_driver_sql("SELECT * FROM sales LIMIT 5")
        sample_data = data_result.fetchall()

    # Format schema details
    schema_info = "\n".join([
        f"Column Name: {col[1]}, Data Type: {col[2]}"
        for col in columns
    ])

    # Format sample data
    sample_data_str = "\n".join([str(row) for row in sample_data])

    # Combine schema and sample data
    full_info = f"Schema Information:\n{schema_info}\n\nSample Data:\n{sample_data_str}"

    return full_info
# Example Query: Get all sales in the "Electronics" category from the "North" region
# query = """
# SELECT * FROM sales 
# WHERE "Product Category" = 'Electronics' 
# AND Region = 'North'
# """

# Run query
# result_df = query_sales_db(query, engine)
# print(result_df.head())  # Display results
# load_and_enrich_data()