import os
import pandas as pd
from pathlib import Path
import re
from chardet import detect

# Load the mapping file with proper encoding
mapping_df = pd.read_csv('/home/ryuu/Ryu/Dataset_Analysis/CEJC_Analyse/Speaker_Conversation_Mapping_Valid.csv', encoding='shift_jis')

# Directory containing cleaned data
cleaned_data_dir = Path('/home/ryuu/Ryu/Dataset_Analysis/CEJC_Analyse/cleaned-data')

def detect_encoding(file_path):
    """Detect file encoding"""
    try:
        with open(file_path, 'rb') as f:
            result = detect(f.read())
        return result['encoding'] or 'utf-8'
    except:
        return 'utf-8'

def count_mora(text):
    """
    Count mora (モーラ) in Japanese text.
    Mora includes: hiragana, katakana, and kanji combinations.
    """
    if not isinstance(text, str):
        return 0
    
    # Count hiragana and katakana characters
    hiragana_count = len(re.findall(r'[\u3040-\u309F]', text))
    katakana_count = len(re.findall(r'[\u30A0-\u30FF]', text))
    
    # For kanji, count as 1 mora per kanji
    kanji_count = len(re.findall(r'[\u4E00-\u9FFF]', text))
    
    total_mora = hiragana_count + katakana_count + kanji_count
    return total_mora

# Dictionary to store mora counts by core_speaker
mora_by_speaker = {}
detailed_results = []
gender_pair_results = []

# Process each row in the mapping file
for idx, row in mapping_df.iterrows():
    session_id = row['session_id']
    conversation_id = row['conversation_id']
    core_speaker = row['core_speaker']
    csv_file = row['csv_file']
    core_speaker_age = row['core_speaker_age']
    core_speaker_gender = row['core_speaker_gender']
    all_speakers_detailed = row['all_speakers_detailed']
    location = row['location']
    relationship = row['relationship']
    
    # Initialize speaker if not exists
    if core_speaker not in mora_by_speaker:
        mora_by_speaker[core_speaker] = {
            'total_mora': 0,
            'num_conversations': 0,
            'num_utterances': 0,
            'total_duration': 0.0,
            'age': core_speaker_age,
            'gender': core_speaker_gender,
            'details': []
        }
    
    # Construct path to the conversation csv file
    conv_dir = cleaned_data_dir / session_id[:4] / conversation_id
    csv_path = conv_dir / csv_file
    
    if csv_path.exists():
        try:
            # Detect encoding and read the conversation file
            encoding = detect_encoding(csv_path)
            conv_df = pd.read_csv(csv_path, encoding=encoding)
            
            # Debug: Print column names for first file
            if idx == 0:
                print(f"Column names in {csv_path}: {list(conv_df.columns)}")
                print(f"First few rows:\n{conv_df.head()}\n")
            
            # Check for correct column names
            if 'speakerID' not in conv_df.columns or 'text' not in conv_df.columns:
                print(f"Warning: Expected 'speakerID' and 'text' columns in {csv_path}")
                print(f"Available columns: {list(conv_df.columns)}")
                continue
            
            # Extract speaker name from speakerID (format: "IC01_玲子")
            # Filter for current core speaker's utterances
            speaker_data = conv_df[conv_df['speakerID'].str.contains(core_speaker, na=False)]
            
            # Count mora for each utterance by the core speaker
            conversation_mora = 0
            conversation_utterances = 0
            conversation_duration = 0.0
            
            for idx_utt, utterance_row in speaker_data.iterrows():
                utterance_text = utterance_row['text']
                utterance_mora = count_mora(utterance_text)
                conversation_mora += utterance_mora
                conversation_utterances += 1
                
                # Calculate duration if endTime and startTime columns exist
                if 'endTime' in conv_df.columns and 'startTime' in conv_df.columns:
                    try:
                        start_time = float(utterance_row['startTime'])
                        end_time = float(utterance_row['endTime'])
                        conversation_duration += (end_time - start_time)
                    except:
                        pass
            
            # Update speaker statistics
            mora_by_speaker[core_speaker]['total_mora'] += conversation_mora
            mora_by_speaker[core_speaker]['num_conversations'] += 1
            mora_by_speaker[core_speaker]['num_utterances'] += conversation_utterances
            mora_by_speaker[core_speaker]['total_duration'] += conversation_duration
            mora_by_speaker[core_speaker]['details'].append({
                'conversation_id': conversation_id,
                'mora': conversation_mora,
                'utterances': conversation_utterances,
                'duration': conversation_duration
            })
            
            # Add to detailed results
            detailed_results.append({
                'core_speaker': core_speaker,
                'age': core_speaker_age,
                'gender': core_speaker_gender,
                'conversation_id': conversation_id,
                'all_speakers_detailed': all_speakers_detailed,
                'location': location,
                'relationship': relationship,
                'mora_count': conversation_mora,
                'utterances': conversation_utterances,
                'duration_seconds': conversation_duration
            })
            
            # Extract gender pair from all_speakers_detailed
            # Format: "玲子(女性, 40-44歳) | 美沙(女性, 40-44歳)"
            genders = []
            try:
                speaker_infos = all_speakers_detailed.split(' | ')
                for info in speaker_infos:
                    if '女性' in info:
                        genders.append('F')
                    elif '男性' in info:
                        genders.append('M')
            except:
                genders = []
            
            # Determine gender pair
            if len(genders) == 2:
                gender_pair = '-'.join(sorted(genders))  # Normalized pair (e.g., F-M, F-F, M-M)
                gender_pair_results.append({
                    'conversation_id': conversation_id,
                    'core_speaker': core_speaker,
                    'gender_pair': gender_pair,
                    'mora_count': conversation_mora,
                    'utterances': conversation_utterances,
                    'duration_seconds': conversation_duration,
                    'location': location,
                    'relationship': relationship
                })
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
    else:
        print(f"File not found: {csv_path}")

# Create a summary dataframe
summary_data = []
for speaker, stats in mora_by_speaker.items():
    total_mora = stats['total_mora']
    num_conversations = stats['num_conversations']
    num_utterances = stats['num_utterances']
    total_duration = stats['total_duration']
    
    avg_mora_per_conversation = total_mora / num_conversations if num_conversations > 0 else 0
    avg_mora_per_utterance = total_mora / num_utterances if num_utterances > 0 else 0
    mora_per_second = total_mora / total_duration if total_duration > 0 else 0
    
    summary_data.append({
        'core_speaker': speaker,
        'gender': stats['gender'],
        'age': stats['age'],
        'total_mora': total_mora,
        'num_conversations': num_conversations,
        'num_utterances': num_utterances,
        'total_duration_seconds': round(total_duration, 2),
        'avg_mora_per_conversation': round(avg_mora_per_conversation, 2),
        'avg_mora_per_utterance': round(avg_mora_per_utterance, 2),
        'mora_per_second': round(mora_per_second, 2)
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('total_mora', ascending=False)

# Create gender pair analysis
gender_pair_df = pd.DataFrame(gender_pair_results)

# Aggregate by gender pair
gender_pair_summary = []
for pair in ['F-F', 'F-M', 'M-M']:
    pair_data = gender_pair_df[gender_pair_df['gender_pair'] == pair]
    if len(pair_data) > 0:
        total_mora = pair_data['mora_count'].sum()
        num_conversations = len(pair_data)
        num_utterances = pair_data['utterances'].sum()
        total_duration = pair_data['duration_seconds'].sum()
        
        avg_mora_per_conversation = total_mora / num_conversations if num_conversations > 0 else 0
        avg_mora_per_utterance = total_mora / num_utterances if num_utterances > 0 else 0
        mora_per_second = total_mora / total_duration if total_duration > 0 else 0
        
        gender_pair_summary.append({
            'gender_pair': pair,
            'num_conversations': num_conversations,
            'total_mora': total_mora,
            'num_utterances': num_utterances,
            'total_duration_seconds': round(total_duration, 2),
            'avg_mora_per_conversation': round(avg_mora_per_conversation, 2),
            'avg_mora_per_utterance': round(avg_mora_per_utterance, 2),
            'mora_per_second': round(mora_per_second, 2)
        })

gender_pair_summary_df = pd.DataFrame(gender_pair_summary)

# Create detailed dataframe only if we have results
if detailed_results:
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df = detailed_df.sort_values('mora_count', ascending=False)
else:
    detailed_df = pd.DataFrame()

# Display results
print("=" * 120)
print("MORA SPOKEN BY EACH CORE SPEAKER (SUMMARY)")
print("=" * 120)
print(summary_df.to_string(index=False))
print("=" * 120)
print(f"\nTotal speakers analyzed: {len(summary_df)}")

print("\n" + "=" * 100)
print("GENDER PAIR ANALYSIS")
print("=" * 100)
print(gender_pair_summary_df.to_string(index=False))
print("=" * 100)

# Save summary to CSV
summary_output_path = '/home/ryuu/Ryu/Dataset_Analysis/mora_by_speaker_summary.csv'
summary_df.to_csv(summary_output_path, index=False, encoding='utf-8')
print(f"\nSummary saved to: {summary_output_path}")

# Save gender pair summary to CSV
gender_pair_output_path = '/home/ryuu/Ryu/Dataset_Analysis/mora_by_gender_pair_summary.csv'
gender_pair_summary_df.to_csv(gender_pair_output_path, index=False, encoding='utf-8')
print(f"Gender pair summary saved to: {gender_pair_output_path}")

# Save detailed results to CSV
if not detailed_df.empty:
    detailed_output_path = '/home/ryuu/Ryu/Dataset_Analysis/mora_by_speaker_detailed.csv'
    detailed_df.to_csv(detailed_output_path, index=False, encoding='utf-8')
    print(f"Detailed results saved to: {detailed_output_path}")
else:
    print("Warning: No detailed results to save")

# Save gender pair detailed results
if not gender_pair_df.empty:
    gender_pair_detailed_output_path = '/home/ryuu/Ryu/Dataset_Analysis/mora_by_gender_pair_detailed.csv'
    gender_pair_df = gender_pair_df.sort_values('mora_count', ascending=False)
    gender_pair_df.to_csv(gender_pair_detailed_output_path, index=False, encoding='utf-8')
    print(f"Gender pair detailed results saved to: {gender_pair_detailed_output_path}")