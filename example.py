"""
Simple example showing how to use the data analysis agent programmatically.
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import os
from data_analysis_agent import create_data_analysis_agent


def run_example_analysis():
    """
    Run example queries on the sample dataset.
    """
    print("Initializing Data Analysis Agent...")
    agent = create_data_analysis_agent()
    
    # Example queries to demonstrate capabilities
    queries = [
        "Load the file sample_sales_data.csv",
        "What are the column names in the dataset?",
        "What's the total revenue across all products?",
        "Which region has the highest average revenue?",
        "Create a bar chart showing revenue by category",
    ]
    
    print("\n" + "="*60)
    print("Running Example Analysis")
    print("="*60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n[Query {i}/{len(queries)}]: {query}")
        print("-" * 60)
        
        try:
            response = agent.invoke({"input": query})
            print(f"\nAnswer: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*60)
    
    print("\n\nExample complete! Check the current directory for generated plots.")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
        exit(1)
    
    run_example_analysis()
