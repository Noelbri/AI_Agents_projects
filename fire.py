from firecrawl import FirecrawlApp
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import os
import json
load_dotenv()
api_key = os.getenv("FIRECRAWL_API_KEY")
# Initialize with your Firecrawl API key
app = FirecrawlApp(api_key=api_key)

# Define a single product schema
class ProductItem(BaseModel):
    productName: Optional[str] = None
    newPrice: Optional[str] = None
    oldPrice: Optional[str] = None

# Define the list schema for multiple products
class ExtractSchema(BaseModel):
    products: List[ProductItem]

# Convert to JSON schema
schema_dict = ExtractSchema.model_json_schema()

# Perform extraction
datta = app.extract(
    urls=["https://jumia.co.ke/flash-sales"],
    prompt="Fetch all product names, old prices, and new prices listed on offer. Return them as a list under a 'products' key.",
    schema=schema_dict,
    enable_web_search=True
)

if datta.success:
    with open("products.json", "w", encoding="utf-8") as f:
        json.dump(datta.data, f, ensure_ascii=False, indent=4)
    print("Data saved")
else:
    print("Failed", datta.error)