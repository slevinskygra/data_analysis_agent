"""
Data Analysis Agent using Claude and LangChain

This agent can load datasets (CSV/Excel), analyze them, and create visualizations.
It uses Python code execution to perform complex data analysis tasks.
"""

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_experimental.utilities import PythonREPL

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Global variable to store loaded datasets
datasets: Dict[str, pd.DataFrame] = {}


def load_data(file_path: str) -> str:
    """
    Load a CSV or Excel file into memory.
    
    Args:
        file_path: Path to the CSV or Excel file
        
    Returns:
        A summary of the loaded dataset
    """
    try:
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return "Error: File must be CSV or Excel format (.csv, .xlsx, .xls)"
        
        # Store the dataframe globally
        dataset_name = os.path.basename(file_path).split('.')[0]
        datasets[dataset_name] = df
        
        # Create summary
        summary = f"Successfully loaded dataset '{dataset_name}'\n\n"
        summary += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        summary += f"Data types:\n{df.dtypes.to_string()}\n\n"
        summary += f"First few rows:\n{df.head(3).to_string()}\n\n"
        summary += f"Basic statistics:\n{df.describe().to_string()}"
        
        return summary
    
    except Exception as e:
        return f"Error loading file: {str(e)}"


def list_datasets(dummy: str = "") -> str:
    """
    List all currently loaded datasets.
    
    Args:
        dummy: Unused parameter (LangChain passes empty string)
    
    Returns:
        A formatted list of loaded datasets with their shapes
    """
    if not datasets:
        return "No datasets currently loaded. Use load_data tool to load a dataset."
    
    result = "Currently loaded datasets:\n\n"
    for name, df in datasets.items():
        result += f"- {name}: {df.shape[0]} rows × {df.shape[1]} columns\n"
        result += f"  Columns: {', '.join(df.columns.tolist()[:5])}"
        if len(df.columns) > 5:
            result += f" ... and {len(df.columns) - 5} more"
        result += "\n\n"
    
    return result


def get_dataset_info(dataset_name: str) -> str:
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to get info about
        
    Returns:
        Detailed information about the dataset
    """
    if dataset_name not in datasets:
        return f"Dataset '{dataset_name}' not found. Available datasets: {', '.join(datasets.keys())}"
    
    df = datasets[dataset_name]
    
    info = f"Dataset: {dataset_name}\n"
    info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
    info += f"Columns and types:\n{df.dtypes.to_string()}\n\n"
    info += f"Missing values:\n{df.isnull().sum().to_string()}\n\n"
    info += f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
    info += f"Sample data:\n{df.head().to_string()}"
    
    return info


def python_repl_tool(code: str) -> str:
    """
    Execute Python code for data analysis.
    The code has access to all loaded datasets via the 'datasets' dictionary.
    
    Args:
        code: Python code to execute
        
    Returns:
        Output from the code execution
    """
    try:
        # Create a Python REPL with access to loaded datasets
        repl = PythonREPL()
        
        # Make datasets available in the execution environment
        setup_code = "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport numpy as np\n"
        setup_code += f"datasets = {repr({k: None for k in datasets.keys()})}\n"
        
        # Load actual dataframes
        for name, df in datasets.items():
            setup_code += f"datasets['{name}'] = pd.read_csv('temp_{name}.csv')\n"
            df.to_csv(f'temp_{name}.csv', index=False)
        
        # Execute setup and user code
        full_code = setup_code + "\n" + code
        result = repl.run(full_code)
        
        # Clean up temp files
        for name in datasets.keys():
            temp_file = f'temp_{name}.csv'
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return str(result) if result else "Code executed successfully (no output)"
    
    except Exception as e:
        return f"Error executing code: {str(e)}"


def create_visualization(dataset_name: str, plot_type: str, x_column: str = None, 
                        y_column: str = None, title: str = None) -> str:
    """
    Create a visualization from a dataset.
    
    Args:
        dataset_name: Name of the dataset to visualize
        plot_type: Type of plot (histogram, scatter, bar, line, box)
        x_column: Column for x-axis (required for most plots)
        y_column: Column for y-axis (required for scatter, line)
        title: Optional title for the plot
        
    Returns:
        Path to the saved plot or error message
    """
    try:
        if dataset_name not in datasets:
            return f"Dataset '{dataset_name}' not found. Available datasets: {', '.join(datasets.keys())}"
        
        df = datasets[dataset_name]
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == "histogram":
            if not x_column:
                return "Error: x_column is required for histogram"
            plt.hist(df[x_column].dropna(), bins=30, edgecolor='black')
            plt.xlabel(x_column)
            plt.ylabel('Frequency')
            
        elif plot_type == "scatter":
            if not x_column or not y_column:
                return "Error: both x_column and y_column are required for scatter plot"
            plt.scatter(df[x_column], df[y_column], alpha=0.6)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        elif plot_type == "bar":
            if not x_column:
                return "Error: x_column is required for bar plot"
            if y_column:
                df.groupby(x_column)[y_column].mean().plot(kind='bar')
                plt.ylabel(f'Average {y_column}')
            else:
                df[x_column].value_counts().plot(kind='bar')
                plt.ylabel('Count')
            plt.xlabel(x_column)
            plt.xticks(rotation=45, ha='right')
            
        elif plot_type == "line":
            if not x_column or not y_column:
                return "Error: both x_column and y_column are required for line plot"
            df_sorted = df.sort_values(x_column)
            plt.plot(df_sorted[x_column], df_sorted[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        elif plot_type == "box":
            if not y_column:
                return "Error: y_column is required for box plot"
            if x_column:
                df.boxplot(column=y_column, by=x_column)
                plt.xlabel(x_column)
            else:
                plt.boxplot(df[y_column].dropna())
            plt.ylabel(y_column)
            
        else:
            return f"Error: Unknown plot type '{plot_type}'. Available types: histogram, scatter, bar, line, box"
        
        if title:
            plt.title(title)
        else:
            plt.title(f'{plot_type.capitalize()} Plot')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = f'plot_{dataset_name}_{plot_type}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"Visualization saved to: {output_file}"
    
    except Exception as e:
        return f"Error creating visualization: {str(e)}"


def create_visualization_wrapper(input_str: str) -> str:
    """
    Wrapper for create_visualization that parses comma-separated input.
    
    Args:
        input_str: Comma-separated values: dataset_name,plot_type,x_column,y_column,title
        
    Returns:
        Result from create_visualization
    """
    try:
        # Parse the input
        parts = [p.strip() for p in input_str.split(',')]
        
        if len(parts) < 2:
            return "Error: Need at least dataset_name and plot_type. Format: dataset_name,plot_type,x_column,y_column,title"
        
        dataset_name = parts[0]
        plot_type = parts[1]
        x_column = parts[2] if len(parts) > 2 and parts[2] else None
        y_column = parts[3] if len(parts) > 3 and parts[3] else None
        title = parts[4] if len(parts) > 4 and parts[4] else None
        
        return create_visualization(dataset_name, plot_type, x_column, y_column, title)
    
    except Exception as e:
        return f"Error parsing input: {str(e)}. Expected format: dataset_name,plot_type,x_column,y_column,title"


def create_data_analysis_agent():
    """
    Create and configure the data analysis agent.
    
    Returns:
        AgentExecutor: The configured agent ready to use
    """
    # Initialize Claude model
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Define the tools the agent can use
    tools = [
        Tool(
            name="load_data",
            func=load_data,
            description=(
                "Load a CSV or Excel file into memory for analysis. "
                "Input should be the file path. "
                "Returns a summary of the loaded dataset including columns, shape, and sample data."
            )
        ),
        Tool(
            name="list_datasets",
            func=list_datasets,
            description=(
                "List all currently loaded datasets with their basic information. "
                "No input required."
            )
        ),
        Tool(
            name="get_dataset_info",
            func=get_dataset_info,
            description=(
                "Get detailed information about a specific loaded dataset. "
                "Input should be the dataset name. "
                "Returns detailed info including columns, types, missing values, and sample data."
            )
        ),
        Tool(
            name="python_repl",
            func=python_repl_tool,
            description=(
                "Execute Python code to perform complex data analysis. "
                "You have access to pandas, matplotlib, seaborn, and numpy. "
                "Loaded datasets are available in a 'datasets' dictionary. "
                "Example: datasets['mydata'].groupby('category')['value'].mean() "
                "Use this for calculations, filtering, grouping, and complex analysis."
            )
        ),
        Tool(
            name="create_visualization",
            func=create_visualization_wrapper,
            description=(
                "Create a visualization from a loaded dataset. "
                "Input should be comma-separated values: dataset_name, plot_type, x_column, y_column, title. "
                "Plot types: histogram, scatter, bar, line, box. "
                "Example input: sample_sales_data,scatter,price,quantity_sold,Price vs Quantity"
            )
        ),
    ]
    
    # Get the ReAct prompt template
    prompt = hub.pull("hwchase17/react")
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create an executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor


def main():
    """
    Main function to run the data analysis assistant.
    """
    print("=" * 60)
    print("Data Analysis Agent with Claude")
    print("=" * 60)
    print("\nThis agent can load datasets, analyze them, and create visualizations.")
    print("Supported formats: CSV, Excel (.xlsx, .xls)")
    print("\nExample queries:")
    print("  - Load the file sales_data.csv")
    print("  - What are the top 5 products by revenue?")
    print("  - Create a scatter plot of price vs quantity")
    print("  - Calculate the average sales by region")
    print("\nType 'quit' or 'exit' to stop.\n")
    
    # Create the agent
    try:
        agent = create_data_analysis_agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("\nMake sure you have:")
        print("1. Created a .env file with your ANTHROPIC_API_KEY")
        print("2. Installed all requirements: pip install -r requirements.txt")
        return
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Run the agent
            print("\nAgent: ", end="", flush=True)
            response = agent.invoke({"input": user_input})
            
            # Print the final answer
            print(f"\n{response['output']}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
