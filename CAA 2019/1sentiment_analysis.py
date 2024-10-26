import os
from textblob import TextBlob
import pandas as pd

# Define the path to your articles
path = r'C:\Users\92317\Pictures\CAA 2019'

# Define keywords for each factor
keywords = {
    'Public Sentiment': ['discrimination', 'fear', 'anxiety', 'resentment', 'support', 'opposition', 'controversial', 'acceptance', 'criticism', 'reaction', 'satisfaction', 'discontent', 'backlash', 'divisive', 'empathy'],
    'Media Coverage': ['reportage', 'headlines', 'articles', 'press', 'news', 'coverage', 'broadcasting', 'analysis', 'stories', 'publications', 'updates', 'journalism', 'media outlets', 'reports', 'narratives'],
    'Government Statements and Actions': ['policy', 'legislation', 'amendment', 'official', 'decree', 'ruling', 'statement', 'government', 'announcement', 'action', 'directive', 'implementation', 'intervention', 'response', 'position', 'proclamation'],
    'Legal Challenges and Court Rulings': ['lawsuit', 'litigation', 'judgment', 'ruling', 'case', 'tribunal', 'appeal', 'verdict', 'legal action', 'court', 'international court', 'human rights', 'UN', 'dispute', 'challenge', 'claim', 'legal precedent', 'intervention'],
    'Public Protests and Demonstrations': ['rally', 'protest', 'march', 'demonstration', 'sit-in', 'strike', 'activism', 'civil disobedience', 'outcry', 'mobilization', 'assembly', 'public meeting', 'unrest', 'resistance', 'grievance']
}

def analyze_article(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    keyword_analysis = {factor: sum(text.lower().count(keyword) for keyword in keywords_list) 
                        for factor, keywords_list in keywords.items()}
    
    return polarity, subjectivity, keyword_analysis

# Create a list to hold the results
results = []

# Analyze each article
for i in range(1, 22):
    file_path = os.path.join(path, f'article {i}.txt')
    if os.path.isfile(file_path):
        polarity, subjectivity, keyword_analysis = analyze_article(file_path)
        results.append({
            'Article': i,
            'Polarity': polarity,
            'Subjectivity': subjectivity,
            **keyword_analysis
        })

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Save results to a CSV file
df.to_csv('sentiment_analysis_results.csv', index=False)

print("Sentiment analysis completed and results saved to 'sentiment_analysis_results.csv'")
