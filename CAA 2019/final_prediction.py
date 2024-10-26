import pandas as pd
from textblob import TextBlob
import os

# Define the path to your articles
path = 'C:\\Users\\92317\\Pictures\\CAA 2019'

# Keywords for each factor
keywords = {
    'Public Sentiment': ['discrimination', 'fear', 'anxiety', 'resentment', 'support', 'opposition', 'controversial', 'acceptance', 'criticism', 'reaction', 'satisfaction', 'discontent', 'backlash', 'divisive', 'empathy'],
    'Media Coverage': ['reportage', 'headlines', 'articles', 'press', 'news', 'coverage', 'broadcasting', 'analysis', 'stories', 'publications', 'updates', 'journalism', 'media outlets', 'reports', 'narratives'],
    'Government Statements and Actions': ['policy', 'legislation', 'amendment', 'official', 'decree', 'ruling', 'statement', 'government', 'announcement', 'action', 'directive', 'implementation', 'intervention', 'response', 'position', 'proclamation'],
    'Legal Challenges and Court Rulings': ['lawsuit', 'litigation', 'judgment', 'ruling', 'case', 'tribunal', 'appeal', 'verdict', 'legal action', 'court', 'international court', 'human rights', 'UN', 'dispute', 'challenge', 'claim', 'legal precedent', 'intervention'],
    'Public Protests and Demonstrations': ['rally', 'protest', 'march', 'demonstration', 'sit-in', 'strike', 'activism', 'civil disobedience', 'outcry', 'mobilization', 'assembly', 'public meeting', 'unrest', 'resistance', 'grievance']
}

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Initialize results DataFrame
columns = ['Article', 'Polarity', 'Subjectivity', 'Public Sentiment', 'Media Coverage', 'Government Statements and Actions', 'Legal Challenges and Court Rulings', 'Public Protests and Demonstrations']
results = pd.DataFrame(columns=columns)

# Process each article
data_frames = []
for i in range(1, 22):
    file_path = os.path.join(path, f'article {i}.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Analyze sentiment
    polarity, subjectivity = analyze_sentiment(text)

    # Count keywords occurrences for each factor
    keyword_counts = {factor: sum(word in text.lower() for word in words) for factor, words in keywords.items()}

    # Create a DataFrame for the current article
    df = pd.DataFrame([[f'article {i}', polarity, subjectivity] + list(keyword_counts.values())], columns=columns)
    data_frames.append(df)

# Concatenate all DataFrames
results = pd.concat(data_frames, ignore_index=True)

# Save results to CSV
output_path = 'C:\\Users\\92317\\Documents\\sentiment_analysis_results.csv'
results.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")

# Correlation analysis
correlation_matrix = results[['Polarity', 'Subjectivity'] + list(keyword_counts.keys())].corr()
print("Correlation matrix:")
print(correlation_matrix)
