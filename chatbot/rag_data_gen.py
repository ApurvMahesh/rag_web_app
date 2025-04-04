import pandas as pd
import random
import numpy as np

# Parameters
num_records = 5000
regions = ['North', 'South', 'East', 'West']
categories = ['Electronics', 'Apparel', 'Groceries']
customer_segments = ['Retail', 'Wholesale']
sales_origins = ['Mobile', 'Desktop']
payment_methods = ['Credit Card', 'Debit Card', 'UPI']
brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD']
item_sizes = ['Small', 'Medium', 'Large']
start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2024-12-31')

# Generate data
data = {
    "Date": [start_date + pd.Timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_records)],
    "Region": [random.choice(regions) for _ in range(num_records)],
    "Product_Category": [random.choice(categories) for _ in range(num_records)],
    "Product_ID": [f"PROD{random.randint(100, 999)}" for _ in range(num_records)],
    "Brand": [random.choice(brands) for _ in range(num_records)],
    "Item_Size": [random.choice(item_sizes) for _ in range(num_records)],
    "Units_Sold": [random.randint(1, 500) for _ in range(num_records)],
    "Unit_Price": [round(random.uniform(5, 500), 2) for _ in range(num_records)],
    "Discount_Offered": [random.choice([0, 5, 10, 15, 20]) for _ in range(num_records)],
    "Marketing_Spend": [random.randint(0, 1000) for _ in range(num_records)],
    "Customer_Segment": [random.choice(customer_segments) for _ in range(num_records)],
    "Sales_Origin": [random.choice(sales_origins) for _ in range(num_records)],
    "Payment_Method": [random.choice(payment_methods) for _ in range(num_records)],
}

# Calculate Total Revenue
df = pd.DataFrame(data)
df["Discount_Factor"] = 1 - (df["Discount_Offered"] / 100)
df["Total_Revenue"] = round(df["Units_Sold"] * df["Unit_Price"] * df["Discount_Factor"], 2)
df.drop(columns=["Discount_Factor"], inplace=True)

# Save to CSV
df.to_csv("synthetic_sales_data_enriched.csv", index=False)
print("Enriched synthetic sales data saved to 'synthetic_sales_data_enriched.csv'")
