import os
import pandas as pd
from pathlib import Path

# Define paths
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent / "CEJC"
CLEANED_DATA_DIR = SCRIPT_DIR / "cleaned-data"

# Create cleaned-data directory if it doesn't exist
CLEANED_DATA_DIR.mkdir(exist_ok=True)

def read_csv_with_fallback(filepath):
    """Try reading CSV with different encodings"""
    encodings = ['shift_jis', 'cp932', 'utf-8', 'latin1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    
    raise ValueError(f"Could not read {filepath} with any supported encoding")

# Load the valid conversation mapping with encoding fallback
MAPPING_FILE = SCRIPT_DIR / "Speaker_Conversation_Mapping_Valid.csv"
valid_mapping = read_csv_with_fallback(MAPPING_FILE)

# Extract valid CSV files from the mapping
valid_csv_files = set(valid_mapping['csv_file'].unique())

# Define backchannels to remove
BACKCHANNELS = {
    'うん。', 'うんうん。','うん うん。','うん うん うん うん。','うん うん うん。', 'うーん。', 'ええ。', 'はい。', 'そう。', 'ねー。', 'ふーん。', 'へー。', 'あー。'
}

def is_backchannel(text):
    """Check if text is a backchannel or standalone (L) marker"""
    if pd.isna(text):
        return False
    text = str(text).strip()
    return text in BACKCHANNELS or text == '(L)'

def clean_and_copy_files():
    """Find, clean, and copy all valid -luu.csv files"""
    copied_count = 0
    skipped_count = 0
    
    # Search for all -luu.csv files in CEJC dataset
    for luu_file in DATASET_DIR.rglob("*-luu.csv"):
        # Extract the filename without path
        filename = luu_file.name
        
        # Check if this file is in the valid mapping
        if filename not in valid_csv_files:
            continue
        
        try:
            # Read the CSV file with encoding fallback
            df = read_csv_with_fallback(luu_file)
            
            # Remove rows where 'text' column is a backchannel or standalone (L)
            df = df[~df['text'].apply(is_backchannel)]
            
            # Create subdirectory structure in cleaned-data
            relative_path = luu_file.relative_to(DATASET_DIR)
            output_file = CLEANED_DATA_DIR / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save cleaned file (use utf-8 for output)
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✓ Cleaned and copied: {relative_path}")
            copied_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {copied_count} files copied, {skipped_count} files skipped")
    print(f"Cleaned files saved to: {CLEANED_DATA_DIR}")

if __name__ == "__main__":
    clean_and_copy_files()