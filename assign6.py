import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

try:
    # Load dataset from a local file
    # Update the path to point to your local CSV file location
    file_path = "C:\\Users\\Devavrat Tapare\\Downloads\\3rd YEAR\\DMDW Lab\\bookstore_transactions.csv"
    df = pd.read_csv(file_path)

    # Display the dataset
    print("Dataset:")
    print(df)

    # Applying the Apriori algorithm
    frequent_itemsets = apriori(df.drop('Transaction_ID', axis=1), min_support=0.2, use_colnames=True)

    # Generating association rules from frequent itemsets
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}. Please install the required module by running: pip install mlxtend")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
