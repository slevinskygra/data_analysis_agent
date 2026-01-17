# Data Analysis Agent

An AI agent that can load datasets, perform analysis, and create visualizations using Claude and LangChain.

## What This Agent Can Do

This agent is like having a data analyst assistant that can:
- **Load data** from CSV and Excel files
- **Analyze data** using pandas (filtering, grouping, statistics)
- **Execute Python code** for complex analysis
- **Create visualizations** (scatter plots, histograms, bar charts, etc.)
- **Answer questions** about your data in natural language

## How It Works

The agent has access to 5 specialized tools:

1. **load_data**: Loads CSV/Excel files into memory
2. **list_datasets**: Shows what datasets are currently loaded
3. **get_dataset_info**: Gets detailed info about a dataset
4. **python_repl**: Executes Python code for complex analysis
5. **create_visualization**: Creates plots and charts

The agent decides which tools to use based on your questions!

## Setup Instructions

### 1. Create Conda Environment

```bash
conda create -n data-analysis-agent python=3.11
conda activate data-analysis-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

### 4. Run the Agent

```bash
python data_analysis_agent.py
```

## Example Usage

### Basic Analysis

```
You: Load the file sample_sales_data.csv

Agent: [Loads data and shows summary]

You: What are the top 3 products by total revenue?

Agent: [Uses python_repl to calculate]
The top 3 products by total revenue are:
1. Laptop: $489,200
2. Monitor: $234,900
3. Chair: $122,400

You: Show me a bar chart of revenue by region

Agent: [Creates visualization]
Visualization saved to: plot_sales_data_bar.png
```

### Advanced Analysis

```
You: Calculate the average quantity sold per product category

Agent: [Executes analysis]
Average quantity sold by category:
- Electronics: 156.3 units
- Furniture: 58.7 units

You: Create a scatter plot showing price vs quantity sold

Agent: [Creates scatter plot]
Visualization saved to: plot_sales_data_scatter.png
```

## Understanding the Code

### Key Components

**Datasets Dictionary**: 
```python
datasets: Dict[str, pd.DataFrame] = {}
```
This global dictionary stores all loaded datasets so they're accessible to all tools.

**Tools**:
- Each tool is a Python function wrapped in a `Tool` object
- The agent reads the tool descriptions to decide when to use them
- Tools can call each other (e.g., python_repl can access loaded datasets)

**Python REPL Tool**:
This is the most powerful tool - it lets the agent write and execute Python code for complex analysis. The agent has access to pandas, numpy, matplotlib, and all loaded datasets.



## Sample Queries to Try

### Data Loading
- "Load the file sample_sales_data.csv"
- "What datasets do I have loaded?"
- "Tell me about the sales_data dataset"

### Analysis Questions
- "What's the average price by category?"
- "Which region has the highest total revenue?"
- "Show me products with quantity sold above 100"
- "Calculate correlation between price and quantity sold"

### Visualizations
- "Create a histogram of prices"
- "Make a scatter plot of price vs quantity"
- "Show me a bar chart of revenue by month"
- "Create a box plot of quantity by region"

### Complex Analysis
- "Find products where revenue exceeds $50,000"
- "Calculate the monthly growth rate for each region"
- "What percentage of total revenue comes from Electronics?"
- "Show me the top 5 products by revenue in the West region"


## Troubleshooting

**"Dataset not found"**: Make sure you've loaded the dataset first using load_data

**"Column not found"**: Check the column names with get_dataset_info

**Code execution errors**: The python_repl tool has access to pandas, numpy, matplotlib, and seaborn - make sure your code uses these libraries

**Plots not showing**: Plots are saved as PNG files in the current directory - look for files like `plot_*.png`



## Resources

- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
- **LangChain Agents**: https://python.langchain.com/docs/modules/agents/
- **Python REPL Tool**: https://python.langchain.com/docs/integrations/tools/python

