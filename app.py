import streamlit as st
import torch
import numpy as np
import json
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel, pipeline
import re
import pandas as pd
import torch.nn as nn
import seaborn as sns
import spacy
from sentence_transformers import SentenceTransformer
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from scipy.stats import gaussian_kde
import gdown
import os

from emojipy import Emoji

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = {
    "positive": "#4CAF50",
    "neutral": "#FFC107",
    "negative": "#F44336",
    "primary": "#2196F3",
    "secondary": "#9C27B0",
    "accent": "#FF9800"
}

GOOGLE_DRIVE_IDS = {
    "best_model.pt": "YOUR_BEST_MODEL_FILE_ID",
    "full_model.pt": "YOUR_FULL_MODEL_FILE_ID",
    "embeddings.pt": "YOUR_EMBEDDINGS_FILE_ID"
}

@st.cache_resource
def download_from_google_drive():
    with st.spinner("Downloading models from Google Drive..."):
        for file_name, file_id in GOOGLE_DRIVE_IDS.items():
            if not os.path.exists(file_name):
                try:
                    url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    gdown.download(url, file_name, quiet=False)
                    st.success(f"Downloaded {file_name}")
                except Exception as e:
                    st.error(f"Failed to download {file_name}: {str(e)}")
                    raise
    return True

class LSTMCRFClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.6, bidirectional=True):
        super(LSTMCRFClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * num_directions, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, x, tags=None):
        batch_size, seq_len, _ = x.size()
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out)
        emissions = self.fc(lstm_out)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(x.device)
        if tags is not None:
            tags = tags.unsqueeze(1)
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        else:
            decoded = self.crf.decode(emissions, mask=mask)
            return torch.tensor([d[0] for d in decoded], device=x.device)

def emoji_to_words(text, emoji_to_word):
    emoji = Emoji()
    shortcoded = emoji.unicode_to_shortcode(text)
    shortcoded = re.sub(r'(:\w+?:)', r' \1 ', shortcoded).strip()
    parts = shortcoded.split()
    result = []
    for part in parts:
        result.append(emoji_to_word.get(part, part))
    temp_text = ' '.join(result)
    return re.sub(r':(\w+):', r'\1', temp_text)

def preprocess_text(text, tokenizer, bert_model, pca, emoji_to_word, final_dict):
    text = text.lower()
    text = emoji_to_words(text, emoji_to_word)
    text = ' '.join([final_dict.get(word, word) for word in text.split()])
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    with torch.no_grad():
        tokens = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        outputs = bert_model(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :].cpu()
    pca_embedding = pca.transform(embedding.numpy())
    features = np.concatenate(([sentiment_score], pca_embedding[0]))
    return torch.tensor(features, dtype=torch.float32)

def classify_sentence(text, model, label_encoder, tokenizer, bert_model, pca, emoji_to_word, final_dict):
    features = preprocess_text(text, tokenizer, bert_model, pca, emoji_to_word, final_dict)
    features = features.unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        predicted = model(features)
        predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0].lower()
        confidence = 1.0
        probs = np.zeros(3)
        probs[predicted.item()] = 1.0
    return predicted_label, confidence, probs

def load_model_and_dependencies():
    download_from_google_drive()
    local_files = ['pca_model.pkl', 'label_encoder.pkl', 'final_acronyms.json', 'emojis_config.json']
    for file_name in local_files:
        if not os.path.exists(file_name):
            st.error(f"Required file {file_name} not found in repository.")
            raise FileNotFoundError(f"{file_name} missing")
    with open('pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open("final_acronyms.json", "r") as file:
        dictionary = json.load(file)
    final_dict = {k.lower(): v.lower() for k, v in dictionary.items()}
    with open("emojis_config.json", "r") as file:
        emoji_to_word = json.load(file)
    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(device)
    bert_model.eval()
    input_size = 257
    hidden_size = 512
    num_classes = 3
    try:
        model = torch.load('full_model.pt', map_location=device)
        model.eval()
    except:
        model = LSTMCRFClassifier(input_size=input_size, hidden_size=hidden_size, 
                                  num_classes=num_classes, dropout=0.6)
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
        model.to(device)
        model.eval()
    return model, label_encoder, tokenizer, bert_model, pca, emoji_to_word, final_dict

def get_sentiment_emoji(label):
    return "😊" if label.lower() == "positive" else "😠" if label.lower() == "negative" else "😐"

def load_language_model(text):
    try:
        lang = detect(text)
        model_name = f"{lang}_core_web_sm" if lang != "en" else "en_core_web_sm"
        try:
            return spacy.load(model_name)
        except OSError:
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
    except:
        return spacy.load("en_core_web_sm")

@lru_cache(maxsize=1)
def load_absa_model():
    return pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")

sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "it", "this", "that", "of", "to",
    "for", "on", "with", "as", "by", "at", "in", "and", "but", "or"
}

def predict_aspects(text, previous_aspects=None):
    nlp = load_language_model(text)
    doc = nlp(text)
    aspects = []
    previous_aspects = previous_aspects or []
    for chunk in doc.noun_chunks:
        if "and" in chunk.text.lower() and "also" not in chunk.text.lower():
            sub_chunks = [c.strip() for c in chunk.text.split(" and ")]
            for sub_chunk in sub_chunks:
                sub_doc = nlp(sub_chunk)
                aspect_tokens = [
                    token.text for token in sub_doc
                    if token.text.lower() not in STOPWORDS
                    and token.pos_ in ["NOUN", "PROPN"]
                    and token.dep_ not in ["det", "poss", "prep", "pron"]
                ]
                aspect = " ".join(aspect_tokens).strip()
                if aspect:
                    aspects.append(aspect)
        else:
            aspect_tokens = [
                token.text for token in chunk
                if token.text.lower() not in STOPWORDS
                and token.pos_ in ["NOUN", "PROPN"]
                and token.dep_ not in ["det", "poss", "prep", "pron"]
            ]
            aspect = " ".join(aspect_tokens).strip()
            if aspect:
                aspects.append(aspect)
    if "it" in [token.text.lower() for token in doc] and previous_aspects and not aspects:
        aspects.append(previous_aspects[-1])
    if not aspects:
        for token in doc:
            if (token.text.lower() not in STOPWORDS and
                token.pos_ in ["NOUN", "PROPN"] and
                token.dep_ not in ["det", "poss", "prep", "pron"]):
                aspects.append(token.text)
    aspects = sorted(list(set(aspects)))
    return merge_similar_aspects(aspects)

def merge_similar_aspects(aspects, threshold=0.9):
    if len(aspects) <= 1:
        return aspects
    aspect_vectors = sentence_model.encode(aspects)
    merged_aspects = []
    used_indices = set()
    for i, aspect1 in enumerate(aspects):
        if i in used_indices:
            continue
        merged_aspects.append(aspect1)
        for j, aspect2 in enumerate(aspects):
            if i != j and j not in used_indices:
                similarity = cosine_similarity(
                    [aspect_vectors[i]], [aspect_vectors[j]]
                )[0][0]
                if similarity > threshold:
                    used_indices.add(j)
        used_indices.add(i)
    return merged_aspects

def classify_sentiment_absa(text, aspect_terms):
    absa_classifier = load_absa_model()
    aspect_sentiments = []
    for aspect in aspect_terms:
        input_text = f"[CLS] {text} [SEP] {aspect} [SEP]"
        result = absa_classifier(input_text)
        sentiment = result[0]["label"].lower() if result else "neutral"
        confidence = round(result[0]["score"], 4) if result else 0.0
        aspect_sentiments.append((aspect, sentiment, confidence, text))
    return aspect_sentiments

def split_sentences(text):
    nlp = load_language_model(text)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    final_sentences = []
    for sent in sentences:
        if " but " in sent:
            parts = sent.split(" but ")
            final_sentences.extend(parts)
        elif ",but " in sent:
            parts = sent.split(",but ")
            final_sentences.extend(parts)
        elif " and also " in sent:
            parts = sent.split(" and also ")
            final_sentences.extend(parts)
        else:
            final_sentences.append(sent)
    final_sentences = [s.strip() for s in final_sentences if s.strip()]
    return final_sentences

def get_aspect_sentiments(text):
    clauses = split_sentences(text)
    aspect_sentiment_dict = {}
    previous_aspects = []
    for clause in clauses:
        aspect_terms = predict_aspects(clause, previous_aspects)
        if aspect_terms:
            sentiments = classify_sentiment_absa(clause, aspect_terms)
            for aspect, sentiment, confidence, clause_text in sentiments:
                if aspect not in aspect_sentiment_dict:
                    aspect_sentiment_dict[aspect] = []
                aspect_sentiment_dict[aspect].append((sentiment, confidence, clause_text))
            previous_aspects = aspect_terms
    aspect_sentiments = []
    for aspect, sentiment_list in aspect_sentiment_dict.items():
        for sentiment, confidence, clause_text in sentiment_list:
            aspect_sentiments.append((aspect, sentiment, confidence, clause_text))
    return aspect_sentiments

def get_word_level_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    nlp = load_language_model(text)
    doc = nlp(text)
    word_sentiments = []
    for token in doc:
        if token.text.lower() not in STOPWORDS and token.pos_ in ["NOUN", "ADJ", "VERB", "ADV"]:
            scores = analyzer.polarity_scores(token.text)
            word_sentiments.append({
                "Word": token.text,
                "Compound": scores["compound"],
                "Positive": scores["pos"],
                "Negative": scores["neg"],
                "Neutral": scores["neu"]
            })
    return pd.DataFrame(word_sentiments)

def process_batch_data(uploaded_file, model_dependencies, text_column):
    model, label_encoder, tokenizer, bert_model, pca, emoji_to_word, final_dict = model_dependencies
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.DataFrame({'text': [line.decode('utf-8').strip() for line in uploaded_file if line.strip()]})
    else:
        st.error("Unsupported file format.")
        return None
    if text_column not in df.columns:
        st.error(f"Selected column '{text_column}' not found in file.")
        return None
    df = df[[text_column]].rename(columns={text_column: 'text'})
    results = []
    progress_bar = st.progress(0)
    for idx, row in df.iterrows():
        text = str(row['text'])
        if text.strip():
            label, confidence, probs = classify_sentence(
                text, model, label_encoder, tokenizer, 
                bert_model, pca, emoji_to_word, final_dict
            )
            results.append({
                'review': text,
                'sentiment': label,
                'confidence': confidence,
                'positive_prob': probs[2],
                'neutral_prob': probs[1],
                'negative_prob': probs[0]
            })
        progress_bar.progress((idx + 1) / len(df))
    progress_bar.empty()
    return pd.DataFrame(results)

def display_aspect_sentiments(aspect_sentiments, tab):
    if not aspect_sentiments:
        tab.warning("No aspects were detected in the text.")
        return
    data = []
    for aspect, sentiment, confidence, clause in aspect_sentiments:
        data.append({
            "Aspect": aspect,
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Source Text": clause
        })
    df = pd.DataFrame(data)
    tab.subheader("Filter Aspects")
    all_aspects = sorted(df["Aspect"].unique())
    selected_aspects = tab.multiselect(
        "Select Aspects to Display",
        options=all_aspects,
        default=all_aspects,
        key="aspect_filter"
    )
    filtered_df = df[df["Aspect"].isin(selected_aspects)]
    if filtered_df.empty:
        tab.warning("No data for selected aspects.")
        return
    tab.subheader("Detected Aspects and Sentiments")
    tab.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False)
    tab.download_button(
        label="Download Aspect Data as CSV",
        data=csv,
        file_name="aspect_sentiment_analysis.csv",
        mime="text/csv"
    )

def display_batch_results(results_df, tab):
    if results_df is None or results_df.empty:
        tab.warning("No results to display.")
        return
    tab.subheader("Filter Sentiments")
    all_sentiments = sorted(results_df["sentiment"].unique())
    selected_sentiments = tab.multiselect(
        "Select Sentiments to Display",
        options=all_sentiments,
        default=all_sentiments,
        key="sentiment_filter"
    )
    filtered_df = results_df[results_df["sentiment"].isin(selected_sentiments)]
    if filtered_df.empty:
        tab.warning("No data for selected sentiments.")
        return
    tab.subheader("Batch Analysis Results")
    tab.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False)
    tab.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv"
    )

def main():
    st.set_page_config(page_title="Advanced Sentiment Analysis", page_icon="📊", layout="wide")
    st.sidebar.title("Sentiment Analysis Dashboard")
    st.sidebar.markdown("Analyze sentiments, aspects, or batch data with advanced visualizations.")
    config_path = ".streamlit/config.toml"
    if os.path.exists(config_path):
        st.sidebar.success("Streamlit config loaded successfully.")
    else:
        st.sidebar.warning("Streamlit config (.streamlit/config.toml) not found. This may cause PyTorch errors.")
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.1)
    @st.cache_resource
    def load_cached_models():
        with st.spinner("Loading models..."):
            return load_model_and_dependencies()
    model_dependencies = load_cached_models()
    plt.clf()
    tab1, tab2, tab3 = st.tabs(["General Sentiment Analysis", "Aspect-Based Analysis", "Batch Processing"])
    with tab1:
        tab1.header("🔍 General Sentiment Analysis")
        tab1.markdown("Enter text to analyze overall sentiment, word-level sentiment, and view detailed visualizations.")
        text_input = tab1.text_area("Enter text:", height=150, key="tab1_text")
        if text_input and tab1.button("Analyze Sentiment", key="analyze_sentiment_tab1"):
            with st.spinner("Analyzing sentiment..."):
                label, confidence, probs = classify_sentence(text_input, *model_dependencies)
                analyzer = SentimentIntensityAnalyzer()
                vader_scores = analyzer.polarity_scores(text_input)
                word_sentiments_df = get_word_level_sentiment(text_input)
                tab1.subheader("Overall Sentiment")
                sentiment_emoji = get_sentiment_emoji(label)
                tab1.metric(
                    label="Predicted Sentiment",
                    value=f"{label.title()} {sentiment_emoji}",
                    delta=f"Confidence: {confidence:.2%}"
                )
                if not word_sentiments_df.empty:
                    tab1.subheader("Word-Level Sentiment Analysis")
                    tab1.markdown("Sentiment scores for significant words (nouns, adjectives, verbs, adverbs).")
                    tab1.dataframe(word_sentiments_df[["Word", "Compound", "Positive", "Negative", "Neutral"]])
                    csv = word_sentiments_df.to_csv(index=False)
                    tab1.download_button(
                        label="Download Word-Level Sentiment as CSV",
                        data=csv,
                        file_name="word_level_sentiment.csv",
                        mime="text/csv"
                    )
    with tab2:
        tab2.header("🔬 Aspect-Based Sentiment Analysis")
        tab2.markdown("Enter text to detect aspects and their sentiments with interactive charts.")
        absa_text_input = tab2.text_area("Enter text for aspect analysis:", height=150, key="tab2_text")
        if absa_text_input and tab2.button("Analyze Aspects", key="analyze_aspects_tab2"):
            with st.spinner("Analyzing aspects and sentiments..."):
                aspect_sentiments = get_aspect_sentiments(absa_text_input)
                display_aspect_sentiments(aspect_sentiments, tab2)
    with tab3:
        tab3.header("📦 Batch Sentiment Analysis")
        tab3.markdown("Upload a file to process multiple reviews and explore aggregated results.")
        uploaded_file = tab3.file_uploader("Upload file", type=["csv", "xlsx", "xls", "txt"], key="tab3_upload")
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                preview_df = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                preview_df = pd.read_excel(uploaded_file)
                uploaded_file.seek(0)
            else:
                preview_df = pd.DataFrame({'text': [line.decode('utf-8').strip() for line in uploaded_file if line.strip()]})
                uploaded_file.seek(0)
            tab3.subheader("Data Preview")
            tab3.dataframe(preview_df.head())
            text_column = tab3.selectbox(
                "Select the column containing reviews:",
                options=preview_df.columns,
                key="tab3_select"
            )
            if tab3.button("Process Batch", key="process_batch_tab3"):
                with st.spinner("Processing batch data..."):
                    results_df = process_batch_data(uploaded_file, model_dependencies, text_column)
                    if results_df is not None and not results_df.empty:
                        display_batch_results(results_df, tab3)

if __name__ == "__main__":
    main()
