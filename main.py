import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
import re
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load summarization model
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text
def summarize_text(text, max_length=500):
    if len(text) > 1024:  # Transformers models have a limit, so split text
        text = text[:1024]
    summary = summarization_pipeline(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to extract keywords
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    keywords = [word for word in words if word not in stop_words and len(word) > 1]

    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform([' '.join(keywords)])
    vocabulary = vectorizer.vocabulary_
    top_keywords = sorted(vocabulary, key=vocabulary.get, reverse=True)[:5]

    return top_keywords

# Function for topic modeling
def topic_modeling(text):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = vectorizer.fit_transform([text])
    
    if tf.shape[1] == 0:
        return ["Not enough unique words for topic modeling."]
    
    lda_model = LatentDirichletAllocation(n_components=3, max_iter=5, learning_method='online', random_state=42)
    lda_model.fit(tf)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic in lda_model.components_:
        topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
    
    return topics

# Extract YouTube video ID
def extract_video_id(url):
    patterns = [
        r'v=([^&]+)',  
        r'youtu.be/([^?]+)',  
        r'youtube.com/embed/([^?]+)',  
        r'/v/([^?]+)'  
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Streamlit app
def main():
    st.title("YouTube Video Summarizer")

    video_url = st.text_input("Enter YouTube Video URL:", "")
    max_summary_length = st.slider("Max Summary Length:", 100, 500, 500)

    if st.button("Summarize"):
        try:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL.")
                return

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                st.error("Transcript not available.")
                return

            video_text = ' '.join([line['text'] for line in transcript])

            summary = summarize_text(video_text, max_length=max_summary_length)
            keywords = extract_keywords(video_text)
            topics = topic_modeling(video_text)
            sentiment = TextBlob(video_text).sentiment

            st.subheader("Summary:")
            st.write(summary)

            st.subheader("Keywords:")
            st.write(keywords)

            st.subheader("Topics:")
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx+1}: {', '.join(topic)}")

            st.subheader("Sentiment Analysis:")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            st.error("No transcript found for this video.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
