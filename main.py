import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from wordcloud import WordCloud
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation

# Losowe dane tekstowe z Quotable API
response = requests.get('https://api.quotable.io/quotes?limit=20')
data = response.json()
text_data = [quote['content'] for quote in data['results']]
df = pd.DataFrame({'text': text_data})

# Top 10 reprezentacji słów

# Przekształcenie danych tekstowych w macierz liczby tokenów
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Całkowita liczba poszczególnych słów
word_counts = np.asarray(X.sum(axis=0)).flatten()
vocab = vectorizer.get_feature_names_out()
word_freq = pd.DataFrame({'word': vocab, 'count': word_counts})
top_10_words = word_freq.sort_values(by='count', ascending=False).head(10)

# Diagram 10 najczęściej używanych słów
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_words, x='word', y='count')
plt.title('Diagram 10 najczęściej używanych słów')
plt.xlabel('Słowo')
plt.ylabel('Częstotliwość')
plt.show()

# Chmura słów

# Generacja chumury słów
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))

# Wykres
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Chmura słów')
plt.show()

# Analiza emocji / wydźwięku

# Analiza sentymentu
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Wykres
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], bins=10, kde=True)
plt.title('Analiza sentymentu')
plt.xlabel('Polaryzacja')
plt.ylabel('Częstotliwość występowania')
plt.show()

# Analiza np. 3 tematów ze zbioru słów lub inna analiza

# LDA topic modeling
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

# Display the top words for each topic
n_top_words = 5
vocab = vectorizer.get_feature_names_out()
topics = []

for topic_idx, topic in enumerate(lda.components_):
    top_features = [vocab[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    topics.append(top_features)

# Creating a DataFrame for topics
topics_df = pd.DataFrame(topics, columns=[f'Word {i+1}' for i in range(n_top_words)])
topics_df.index = [f'Topic {i+1}' for i in range(len(topics))]

# Plotting topics
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(lda.components_, columns=vocab), annot=False, cmap='Blues')
plt.title('LDA Topic Modeling')
plt.xlabel('Words')
plt.ylabel('Topics')
plt.show()

# Display topics DataFrame
print(topics_df)
