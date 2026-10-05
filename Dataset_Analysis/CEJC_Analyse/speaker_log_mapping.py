import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

def create_speaker_conversation_mapping(conversation_csv_path, speaker_data_csv_path, speaker_relation_csv_path, output_valid_csv_path, output_excluded_csv_path):
    """
    Creates mapping CSVs showing which speakers appear in which conversations.
    Splits data into valid (2 speakers) and excluded (3+ speakers or unknown age/gender).
    Includes speaker age and gender information.
    
    Args:
        conversation_csv_path: Path to Conversation.csv
        speaker_data_csv_path: Path to Speaker_data.csv
        speaker_relation_csv_path: Path to Speaker_Conversation_Relation.csv
        output_valid_csv_path: Path to save valid mapping (2 speakers only)
        output_excluded_csv_path: Path to save excluded mapping (3+ speakers or incomplete data)
    """
    
    # Read the CSV files
    speaker_data = pd.read_csv(speaker_data_csv_path, encoding='shift_jis')
    speaker_conv = pd.read_csv(speaker_relation_csv_path, encoding='shift_jis')
    conversation_meta = pd.read_csv(conversation_csv_path, encoding='shift_jis')
    
    # Create a mapping of speaker_id to age and gender
    speaker_info = {}
    for _, row in speaker_data.iterrows():
        speaker_id = row['話者ID']
        speaker_info[speaker_id] = {
            'age': row['年齢'],
            'gender': row['性別'],
            'name': row['話者名']
        }
    
    # Create a mapping of conversation_id to session_id and conversation details
    conv_to_session = {}
    for _, row in conversation_meta.iterrows():
        conv_id = row['会話ID']
        conv_to_session[conv_id] = {
            'session_id': row['セッションID'],
            'overview': row['会話概要'],
            'location': row['場所'],
            'relationship': row['話者間の関係性'],
            'num_speakers': row['話者数']
        }
    
    # Group speakers by conversation with their IDs and labels
    conv_speakers = defaultdict(list)
    for _, row in speaker_conv.iterrows():
        conv_id = row['会話ID']
        speaker_id = row['話者ID']
        speaker_label = row['話者ラベル']
        conv_speakers[conv_id].append({
            'speaker_id': speaker_id,
            'speaker_label': speaker_label
        })
    
    # Create the mapping dataframes
    valid_data = []
    excluded_data = []
    
    for conv_id, speakers in conv_speakers.items():
        if conv_id in conv_to_session:
            session_info = conv_to_session[conv_id]
            
            # Extract core speaker (labeled as IC01, N10A, Z10A, etc.)
            core_speaker_info = [s for s in speakers if 'IC01' in s['speaker_label'] or 'N10A' in s['speaker_label'] or 'Z10A' in s['speaker_label']]
            
            if core_speaker_info:
                core_speaker_id = core_speaker_info[0]['speaker_id']
                core_speaker_name = core_speaker_info[0]['speaker_label'].split('_')[1]
            else:
                core_speaker_id = speakers[0]['speaker_id']
                core_speaker_name = speakers[0]['speaker_label'].split('_')[1]
            
            # Build speaker details string with age and gender
            speaker_details = []
            has_unknown_info = False
            for speaker in speakers:
                sp_id = speaker['speaker_id']
                sp_name = speaker['speaker_label'].split('_')[1]
                if sp_id in speaker_info:
                    age = speaker_info[sp_id]['age']
                    gender = speaker_info[sp_id]['gender']
                    # Check if age or gender is unknown (nan, N/A, etc.)
                    if pd.isna(age) or pd.isna(gender) or str(age).lower() == 'nan' or str(gender).lower() == 'nan':
                        has_unknown_info = True
                    speaker_details.append(f"{sp_name}({gender}, {age})")
                else:
                    has_unknown_info = True
                    speaker_details.append(sp_name)
            
            record = {
                'session_id': session_info['session_id'],
                'conversation_id': conv_id,
                'core_speaker': core_speaker_name,
                'core_speaker_age': speaker_info.get(core_speaker_id, {}).get('age', 'N/A'),
                'core_speaker_gender': speaker_info.get(core_speaker_id, {}).get('gender', 'N/A'),
                'all_speakers': ', '.join([s['speaker_label'].split('_')[1] for s in speakers]),
                'all_speakers_detailed': ' | '.join(speaker_details),
                'num_speakers': len(speakers),
                'conversation_overview': session_info['overview'],
                'location': session_info['location'],
                'relationship': session_info['relationship'],
            }
            
            # Split by speaker count and unknown info
            if has_unknown_info:
                record['csv_file'] = f"{conv_id}.csv"
                record['exclude_reason'] = f'Unknown/missing age or gender data'
                excluded_data.append(record)
            elif len(speakers) <= 2:
                record['csv_file'] = f"{conv_id}-luu.csv"
                valid_data.append(record)
            else:
                record['csv_file'] = f"{conv_id}.csv"
                record['exclude_reason'] = f'3+ speakers (対象外): {len(speakers)} speakers'
                excluded_data.append(record)
    
    # Create and sort dataframes
    valid_df = pd.DataFrame(valid_data)
    valid_df = valid_df.sort_values('session_id').reset_index(drop=True)
    
    excluded_df = pd.DataFrame(excluded_data)
    excluded_df = excluded_df.sort_values('session_id').reset_index(drop=True)
    
    # Save to CSV files
    valid_df.to_csv(output_valid_csv_path, index=False, encoding='shift_jis')
    excluded_df.to_csv(output_excluded_csv_path, index=False, encoding='shift_jis')
    
    print(f"✓ Mapping created successfully!")
    print(f"✓ Valid conversations (2 speakers, complete data): {len(valid_df)}")
    print(f"✓ Excluded conversations: {len(excluded_df)}")
    print(f"✓ Saved to: {output_valid_csv_path}")
    print(f"✓ Saved to: {output_excluded_csv_path}")
    print(f"\nFirst 5 valid rows:")
    print(valid_df.head())
    print(f"\nFirst 5 excluded rows:")
    print(excluded_df.head())
    
    return valid_df, excluded_df


if __name__ == "__main__":
    # Get the parent directory (CEJC folder)
    base_path = Path(__file__).parent.parent / "CEJC"
    
    # File paths
    conversation_csv = base_path / "Conversation.csv"
    speaker_data_csv = base_path / "Speaker_data.csv"
    speaker_relation_csv = base_path / "Speaker_Conversation_Relation.csv"
    output_valid_csv = Path(__file__).parent / "Speaker_Conversation_Mapping_Valid.csv"
    output_excluded_csv = Path(__file__).parent / "Speaker_Conversation_Mapping_Excluded.csv"
    
    # Create the mapping
    valid_df, excluded_df = create_speaker_conversation_mapping(
        str(conversation_csv),
        str(speaker_data_csv),
        str(speaker_relation_csv),
        str(output_valid_csv),
        str(output_excluded_csv)
    )