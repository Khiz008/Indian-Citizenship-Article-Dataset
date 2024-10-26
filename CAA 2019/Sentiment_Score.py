import pandas as pd

# Example dataset
data = {
    'Article': list(range(1, 22)),
    'Public Sentiment': [
        5, 3, 3, 1, 8, 6, 6, 7, 9, 2, 5, 4, 6, 4, 0, 2, 1, 8, 4, 5, 51
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate average sentiment score
average_sentiment = df['Public Sentiment'].mean()
print(f'Average Public Sentiment Score: {average_sentiment}')
