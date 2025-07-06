import pandas as pd
import re
# import other libraries as needed for cleaning (e.g., NLTK if doing advanced tokenization)

class ComplaintDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Loads the CSV data into a pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully from {self.data_path}. Rows: {self.df.shape[0]}")
            return self.df
        except FileNotFoundError:
            print(f"Error: The file '{self.data_path}' was not found.")
            return None

    def filter_data(self, products_to_include, narrative_column='Consumer complaint narrative'):
        """Filters the DataFrame by product category and removes rows with empty narratives."""
        if self.df is None:
            print("Data not loaded. Call load_data() first.")
            return None

        print(f"Initial rows before filtering: {self.df.shape[0]}")
        # Filter by product
        filtered_df = self.df[self.df['Product'].isin(products_to_include)].copy()
        print(f"Rows after product filtering: {filtered_df.shape[0]}")

        # Remove records with empty narrative
        filtered_df = filtered_df.dropna(subset=[narrative_column]).copy()
        print(f"Rows after removing empty narratives: {filtered_df.shape[0]}")

        # Select and return only relevant columns
        self.df = filtered_df[['Complaint ID', 'Product', narrative_column]].copy()
        return self.df

    def clean_text(self, text):
        """Cleans a single text string."""
        if pd.isna(text): # Handle potential NaN if not dropped earlier or for safety
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def apply_cleaning(self, narrative_column='Consumer complaint narrative', new_column_name='cleaned_narrative'):
        """Applies the cleaning function to the specified narrative column."""
        if self.df is None:
            print("Data not loaded/filtered. Call load_data() and filter_data() first.")
            return None

        print(f"Applying text cleaning to '{narrative_column}'...")
        self.df[new_column_name] = self.df[narrative_column].apply(self.clean_text)
        print("Text cleaning complete.")
        return self.df

    def save_cleaned_data(self, output_path):
        """Saves the processed DataFrame to a CSV file."""
        if self.df is None:
            print("No data to save. Process data first.")
            return

        self.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

# Example of how this class would be used (e.g., in a separate script or notebook cell)
# if __name__ == "__main__":
#     processor = ComplaintDataProcessor('../data/complaints.csv')
#     processor.load_data()
#     
#     target_products = [
#         'Credit card', 'Personal loan', 'Buy Now, Pay Later (BNPL)',
#         'Savings account', 'Money transfer'
#     ]
#     processor.filter_data(target_products)
#     processor.apply_cleaning()
#     processor.save_cleaned_data('../data/filtered_complaints.csv')