import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import time
import random

# CONFIGURATION
# We fetch slightly more than 300 to account for potential duplicates or empty abstracts
SAMPLES_PER_CLASS = 350 

# arXiv API Categories mapping
# See: https://arxiv.org/category_taxonomy
TOPICS = {
    "Bioinformatics": "cat:q-bio.QM",       # Quantitative Methods (often used for Bioinformatics)
    "Neuroscience": "cat:q-bio.NC",         # Neurons and Cognition
    "Materials Science": "cat:cond-mat.mtrl-sci" # Materials Science
}

def fetch_arxiv_data(topic_name, query_code, max_results):
    """
    Fetches raw XML from arXiv API and parses titles/abstracts.
    """
    base_url = 'http://export.arxiv.org/api/query?'
    # Sort by 'submittedDate' ensures you get a somewhat random/recent slice 
    # rather than the same top 300 papers everyone else gets.
    query = f"search_query={query_code}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
    print(f"Fetching {max_results} papers for {topic_name}...")
    
    try:
        response = urllib.request.urlopen(base_url + query).read()
        root = ET.fromstring(response)
        
        # arXiv XML namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text
            summary = entry.find('atom:summary', ns).text
            
            # CLEANING: arXiv abstracts have lots of \n newlines. Flatten them.
            clean_summary = summary.replace('\n', ' ').strip()
            clean_title = title.replace('\n', ' ').strip()
            
            if len(clean_summary) < 50: # Skip broken/empty abstracts
                continue
                
            papers.append({
                "label": topic_name,
                "text": clean_summary, # Abstract only (as per your strict constraint)
                "title_metadata": clean_title # Keep metadata just in case, but won't train on it
            })
            
        return papers
        
    except Exception as e:
        print(f"Error fetching {topic_name}: {e}")
        return []

def fetch_and_save_data():
    # MAIN EXECUTION
    all_data = []

    for topic, code in TOPICS.items():
        data = fetch_arxiv_data(topic, code, SAMPLES_PER_CLASS)
        all_data.extend(data)
        # Politeness delay to respect arXiv server rules
        time.sleep(3) 

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # FINAL CLEANUP
    # 1. Remove duplicates (crucial for "Unique Dataset" rule)
    df.drop_duplicates(subset=['text'], inplace=True)

    # 2. Shuffle data (so classes aren't ordered 1-2-3)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 3. Check balance
    print("\nDataset Balance:")
    print(df['label'].value_counts())

    # 4. Save
    filename = "./data/raw/scientific_abstracts_dataset.csv"
    df.to_csv(filename, index=False)
    print(f"\nSuccess! Saved {len(df)} abstracts to {filename}")

if __name__ == "__main__":
    fetch_and_save_data()
