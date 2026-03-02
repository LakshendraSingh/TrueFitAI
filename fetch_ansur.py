import os
import sys
import urllib.request
from urllib.parse import urlparse
import pandas as pd

def get_filename(url: str, default: str) -> str:
    """Extracts the filename from the URL, or uses a default if none is found."""
    parsed_path = urlparse(url).path
    filename = os.path.basename(parsed_path)
    return filename if filename else default

def fetch_and_merge(male_url: str, female_url: str, data_dir: str = "data"):
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Extract filenames from the URLs
    male_file = get_filename(male_url, "ANSUR_II_MALE.csv")
    female_file = get_filename(female_url, "ANSUR_II_FEMALE.csv")
    
    male_path = os.path.join(data_dir, male_file)
    female_path = os.path.join(data_dir, female_file)
    merged_path = os.path.join(data_dir, "ansur_ii.csv")
    
    # Download the files
    print(f"Downloading male dataset to: {male_path}")
    urllib.request.urlretrieve(male_url, male_path)
    
    print(f"Downloading female dataset to: {female_path}")
    urllib.request.urlretrieve(female_url, female_path)
    
    # Merge the datasets
    print(f"Merging datasets into: {merged_path}")
    df_m = pd.read_csv(male_path, encoding="latin-1")
    df_f = pd.read_csv(female_path, encoding="latin-1")
    
    combined_df = pd.concat([df_m, df_f], ignore_index=True)
    combined_df.to_csv(merged_path, index=False)
    
    print("Download and merge complete. Original files were kept in the data directory.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_ansur.py <male_url> <female_url>")
        sys.exit(1)
        
    m_url = sys.argv[1]
    f_url = sys.argv[2]
    fetch_and_merge(m_url, f_url)