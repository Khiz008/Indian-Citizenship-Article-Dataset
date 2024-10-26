import os
import pandas as pd
from textblob import TextBlob

# Directory containing the articles
directory = 'C:\\Users\\92317\\Pictures\\CAA 2019'

# List of articles
articles = [f'article {i}.txt' for i in range(1, 22)]

# Keywords for sentiment analysis
keywords = {
    'Public sentiment': ['public opinion', 'public reaction', 'citizen sentiment'],
    'Media coverage': ['news coverage', 'media reports', 'press coverage'],
    'Govt. statements and actions': ['government statements', 'official actions', 'policy statements'],
    'Legal challenges and court rulings': ['court rulings', 'legal challenges', 'judicial decisions', 'international claims'],
    'Public protests and demonstrations': ['protests', 'demonstrations', 'public rallies', 'mass gatherings']
}

# Initialize lists to store results
data = []

# Process each article
for article_name in articles:
    with open(os.path.join(directory, article_name), 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Create TextBlob object
    blob = TextBlob(text)
    
    # Analyze sentiment
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Store results
    data.append({
        'Article': article_name,
        'Polarity': polarity,
        'Subjectivity': subjectivity
    })

# Convert results to DataFrame
df = pd.DataFrame(data)

# Save results to CSV
output_file = 'sentiment_analysis_results.csv'
df.to_csv(output_file, index=False)

print(f'Sentiment analysis results saved to {output_file}')
