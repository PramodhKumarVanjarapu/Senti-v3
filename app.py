import streamlit as st
import torch
import numpy as np
import json
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel, pipeline
from emojipy import Emoji
import re
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sentence_transformers import SentenceTransformer
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from wordcloud import WordCloud
from torchcrf import CRF
from scipy.stats import gaussian_kde
from io import BytesIO
from matplotlib.lines import Line2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define consistent color palette
COLORS = {
    "positive": "#4CAF50",  # Green
    "neutral": "#FFC107",   # Amber
    "negative": "#F44336",  # Red
    "primary": "#2196F3",   # Blue
    "secondary": "#9C27B0", # Purple
    "accent": "#FF9800"     # Orange
}

# LSTMCRFClassifier
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

def generate_sentiment_visualizations(label, confidence, probs, vader_scores, word_sentiments_df, tab):
    tab.subheader("Overall Sentiment Analysis")
    
    col1, col2, col3 = tab.columns(3)

    # Ensure label is lowercase for COLORS dictionary
    label = label.lower()

    # Line Chart for VADER Sentiment Scores
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    sentiment_types = ["Positive", "Neutral", "Negative", "Compound"]
    scores = [vader_scores["pos"], vader_scores["neu"], vader_scores["neg"], vader_scores["compound"]]
    ax1.plot(sentiment_types, scores, marker="o", color=COLORS["primary"], linewidth=2)
    ax1.set_ylim(-1, 1)
    ax1.set_title(f"VADER Sentiment Scores\nSentiment: {label.title()} {get_sentiment_emoji(label)}")
    ax1.set_ylabel("Score")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    fig1.tight_layout()
    col1.pyplot(fig1)
    plt.close()

    # Word-Level Sentiment Heatmap
    if not word_sentiments_df.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        pivot_df = word_sentiments_df.pivot_table(
            index="Word",
            values=["Positive", "Neutral", "Negative"],
            aggfunc="mean"
        ).fillna(0)
        if not pivot_df.empty:
            sns.heatmap(
                pivot_df,
                cmap="RdYlGn",
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                linecolor="lightgray",
                cbar_kws={"label": "Sentiment Score", "shrink": 0.8}
            )
            plt.title("Word-Level Sentiment Scores", fontsize=12, pad=10)
            plt.xlabel("Sentiment", fontsize=10)
            plt.ylabel("Word", fontsize=10)
            fig2.tight_layout()
            col2.pyplot(fig2)
            plt.close()

    # Pie Chart for VADER Sentiment Distribution
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    pie_labels = ["Positive", "Neutral", "Negative"]
    pie_scores = [vader_scores["pos"], vader_scores["neu"], vader_scores["neg"]]
    pie_colors = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]
    ax3.pie(
        pie_scores,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white")
    )
    ax3.set_title("VADER Sentiment Distribution")
    fig3.tight_layout()
    col3.pyplot(fig3)
    plt.close()

    # Display Word-Level Sentiment Table
    if not word_sentiments_df.empty:
        tab.subheader("Word-Level Sentiment Analysis")
        tab.markdown("Sentiment scores for significant words (nouns, adjectives, verbs, adverbs).")
        tab.dataframe(word_sentiments_df[["Word", "Compound", "Positive", "Negative", "Neutral"]])
        
        # Download Word-Level Data
        csv = word_sentiments_df.to_csv(index=False)
        tab.download_button(
            label="Download Word-Level Sentiment as CSV",
            data=csv,
            file_name="word_level_sentiment.csv",
            mime="text/csv"
        )

def create_downloadable_plots(filtered_df, word_sentiments_df, tab, key_suffix, chart_options):
    """Create downloadable high-quality plots"""
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 150
    
    if tab.button(f"Generate High-Quality Visualizations", key=f"gen_viz_{key_suffix}"):
        plots_container = tab.container()
        plots_container.write("Download high-quality versions of the visualizations:")
        
        col1, col2 = plots_container.columns(2)
        figs = []
        
        if "Sentiment Distribution" in chart_options:
            fig1 = plt.figure(figsize=(8, 6))
            sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            plt.pie(
                sentiment_counts["Count"],
                labels=sentiment_counts["Sentiment"],
                colors=[COLORS.get(s.lower(), "#999999") for s in sentiment_counts["Sentiment"]],
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.6, edgecolor='white')
            )
            plt.title("Sentiment Distribution", fontsize=16, pad=20)
            plt.tight_layout()
            fig1.patch.set_alpha(0.0)
            figs.append(("sentiment_distribution", fig1))
        
        if "Word Sentiment Heatmap" in chart_options and not word_sentiments_df.empty:
            fig2 = plt.figure(figsize=(8, max(6, len(word_sentiments_df) * 0.5)))
            pivot_df = word_sentiments_df.pivot_table(
                index="Word",
                values=["Positive", "Neutral", "Negative"],
                aggfunc="mean"
            ).fillna(0)
            if not pivot_df.empty:
                sns.heatmap(
                    pivot_df,
                    cmap="RdYlGn",
                    annot=True,
                    fmt=".2f",
                    linewidths=0.5,
                    linecolor='lightgray',
                    cbar_kws={'label': 'Sentiment Score', 'shrink': 0.8}
                )
                plt.title("Word-Level Sentiment Scores", fontsize=16, pad=20)
                plt.xlabel("Sentiment", fontsize=12)
                plt.ylabel("Word", fontsize=12)
                plt.tight_layout()
                fig2.patch.set_alpha(0.0)
                figs.append(("word_sentiment_heatmap", fig2))
        
        if "VADER Sentiment Line Chart" in chart_options and hasattr(filtered_df, 'vader_scores'):
            fig3 = plt.figure(figsize=(8, 6))
            sentiment_types = ["Positive", "Neutral", "Negative", "Compound"]
            scores = [
                filtered_df.vader_scores.get("pos", 0),
                filtered_df.vader_scores.get("neu", 0),
                filtered_df.vader_scores.get("neg", 0),
                filtered_df.vader_scores.get("compound", 0)
            ]
            plt.plot(sentiment_types, scores, marker="o", color=COLORS["primary"], linewidth=2)
            plt.ylim(-1, 1)
            plt.title("VADER Sentiment Scores", fontsize=16, pad=20)
            plt.ylabel("Score", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig3.patch.set_alpha(0.0)
            figs.append(("vader_sentiment_line", fig3))
        
        if "VADER Sentiment Pie Chart" in chart_options and hasattr(filtered_df, 'vader_scores'):
            fig4 = plt.figure(figsize=(8, 6))
            pie_labels = ["Positive", "Neutral", "Negative"]
            pie_scores = [
                filtered_df.vader_scores.get("pos", 0),
                filtered_df.vader_scores.get("neu", 0),
                filtered_df.vader_scores.get("neg", 0)
            ]
            pie_colors = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]
            plt.pie(
                pie_scores,
                labels=pie_labels,
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(edgecolor="white")
            )
            plt.title("VADER Sentiment Distribution", fontsize=16, pad=20)
            plt.tight_layout()
            fig4.patch.set_alpha(0.0)
            figs.append(("vader_sentiment_pie", fig4))
        
        if "Word Cloud" in chart_options and not word_sentiments_df.empty:
            fig5 = plt.figure(figsize=(8, 6))
            sentiment_weights = word_sentiments_df.set_index("Word")["Compound"].to_dict()
            
            def sentiment_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                sentiment_score = sentiment_weights.get(word, 0)
                if sentiment_score > 0:
                    return COLORS["positive"]
                elif sentiment_score < 0:
                    return COLORS["negative"]
                else:
                    return COLORS["neutral"]
            
            word_freq = word_sentiments_df["Word"].value_counts().to_dict()
            wordcloud_text = " ".join([word + " " * min(int(freq * 10), 50) for word, freq in word_freq.items()])
            
            wordcloud = WordCloud(
                width=600,
                height=300,
                background_color='white',
                color_func=sentiment_color_func,
                prefer_horizontal=0.9,
                max_words=50,
                contour_width=1,
                contour_color='gray',
                collocations=False
            ).generate(wordcloud_text)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title("Sentiment-Based Coloring", fontsize=16, pad=20)
            plt.tight_layout()
            fig5.patch.set_alpha(0.0)
            figs.append(("word_cloud", fig5))
        
        for name, fig in figs:
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
            buf.seek(0)
            col = col1 if name in ["sentiment_distribution", "vader_sentiment_line"] else col2
            col.image(buf, caption=name.replace("_", " ").title(), width=300)
            buf.seek(0)
            col.download_button(
                label=f"Download {name.replace('_', ' ').title()}",
                data=buf,
                file_name=f"{name}.png",
                mime="image/png"
            )
            plt.close(fig)
        
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['figure.dpi'] = 100

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
    
    if len(filtered_df) > 0:
        tab.subheader("Visualization Options")
        chart_options = tab.multiselect(
            "Select which visualizations to show:",
            options=["Sentiment Distribution", "Aspect-Sentiment Heatmap", "Aspect Frequency", "Word Cloud"],
            default=["Sentiment Distribution", "Aspect-Sentiment Heatmap"],
            key="chart_options"
        )
        
        if chart_options:
            tab.subheader("Sentiment Visualization")
            col1, col2 = tab.columns(2)
            
            if "Sentiment Distribution" in chart_options:
                sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                plt.figure(figsize=(6, 4))
                plt.pie(
                    sentiment_counts["Count"],
                    labels=sentiment_counts["Sentiment"],
                    colors=[COLORS.get(s.lower(), "#999999") for s in sentiment_counts["Sentiment"]],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.6, edgecolor='white')
                )
                plt.title("Sentiment Distribution")
                col1.pyplot(plt)
                plt.close()
            
            if "Aspect-Sentiment Heatmap" in chart_options:
                pivot_df = filtered_df.pivot_table(
                    index="Aspect",
                    columns="Sentiment",
                    values="Confidence",
                    aggfunc=np.mean
                ).fillna(0)
                if not pivot_df.empty:
                    pivot_df_normalized = pivot_df.div(pivot_df.sum(axis=1), axis=0).fillna(0)
                    plt.figure(figsize=(6, max(4, len(pivot_df) * 0.5)))
                    ax = sns.heatmap(
                        pivot_df_normalized,
                        cmap="RdYlGn",
                        annot=True,
                        fmt=".2f",
                        linewidths=0.5,
                        linecolor='lightgray',
                        cbar_kws={'label': 'Normalized Confidence', 'shrink': 0.8}
                    )
                    plt.xticks(rotation=0)
                    for _, spine in ax.spines.items():
                        spine.set_visible(True)
                        spine.set_linewidth(0.5)
                        spine.set_color('lightgray')
                    plt.title("Normalized Aspect-Sentiment Confidence", fontsize=12, pad=10)
                    plt.xlabel("Sentiment", fontsize=10)
                    plt.ylabel("Aspect", fontsize=10)
                    col2.pyplot(plt)
                    plt.close()
            
            if "Aspect Frequency" in chart_options:
                aspect_stats = filtered_df.groupby("Aspect").agg({
                    "Confidence": "mean",
                    "Aspect": "count"
                }).rename(columns={"Aspect": "Count"}).reset_index()
                aspect_stats = aspect_stats.sort_values("Count", ascending=False)
                fig, ax1 = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    x="Aspect",
                    y="Count",
                    data=aspect_stats,
                    color=COLORS["positive"],
                    ax=ax1
                )
                ax1.set_ylabel("Count", color=COLORS["positive"])
                ax1.tick_params(axis='y', labelcolor=COLORS["positive"])
                ax2 = ax1.twinx()
                sns.lineplot(
                    x="Aspect",
                    y="Confidence",
                    data=aspect_stats,
                    color=COLORS["primary"],
                    marker="o",
                    ax=ax2
                )
                ax2.set_ylabel("Avg Confidence", color=COLORS["primary"])
                ax2.tick_params(axis='y', labelcolor=COLORS["primary"])
                plt.title("Aspect Frequency and Confidence")
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                col1.pyplot(fig)
                plt.close()
            
            if "Word Cloud" in chart_options:
                aspect_stats = filtered_df.groupby("Aspect").agg({
                    "Confidence": "mean",
                    "Aspect": "count"
                }).rename(columns={"Aspect": "Count"}).reset_index()
                if len(aspect_stats) > 0:
                    sentiment_weights = filtered_df.groupby("Aspect")["Sentiment"].apply(
                        lambda x: sum(1 if s == "positive" else -1 if s == "negative" else 0 for s in x)
                    ).to_dict()
                    
                    wc_color_method = col2.radio(
                        "WordCloud Coloring Method:",
                        ["Sentiment", "Frequency"],
                        key="wc_color_method",
                        horizontal=True
                    )
                    
                    def sentiment_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                        sentiment_score = sentiment_weights.get(word, 0)
                        if sentiment_score > 0:
                            return COLORS["positive"]
                        elif sentiment_score < 0:
                            return COLORS["negative"]
                        else:
                            return COLORS["neutral"]
                    
                    def frequency_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                        intensity = (font_size / 70.0)
                        r = int(COLORS["primary"][1:3], 16)
                        g = int(COLORS["primary"][3:5], 16)
                        b = int(COLORS["primary"][5:7], 16)
                        return f"rgb({r},{g},{int(b * intensity + 100)})"
                    
                    color_func = sentiment_color_func if wc_color_method == "Sentiment" else frequency_color_func
                    
                    aspect_text = " ".join([row["Aspect"] + " " * row["Count"] for _, row in aspect_stats.iterrows()])
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        color_func=color_func,
                        prefer_horizontal=0.9,
                        max_words=50,
                        contour_width=1,
                        contour_color='gray',
                        collocations=False
                    ).generate(aspect_text)
                    
                    plt.figure(figsize=(6, 3))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.title(f"Aspect Word Cloud ({wc_color_method} Based)", pad=15)
                    col2.pyplot(plt)
                    plt.close()
        
        create_downloadable_plots(filtered_df, pd.DataFrame(), tab, "aspect", chart_options)

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
    
    tab.subheader("Results Summary")
    col1, col2 = tab.columns(2)
    
    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    plt.figure(figsize=(6, 4))
    plt.pie(
        sentiment_counts['count'],
        labels=sentiment_counts['sentiment'],
        colors=[COLORS.get(s.lower(), "#999999") for s in sentiment_counts['sentiment']],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor='white')
    )
    plt.title("Sentiment Distribution")
    col1.pyplot(plt)
    plt.close()
    
    plt.figure(figsize=(6, 4))
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in filtered_df['sentiment'].values:
            sns.histplot(
                data=filtered_df[filtered_df['sentiment'] == sentiment],
                x='confidence',
                label=sentiment.title(),
                color=COLORS[sentiment],
                alpha=0.5,
                stat='density'
            )
            if len(filtered_df[filtered_df['sentiment'] == sentiment]) > 1:
                kde = gaussian_kde(filtered_df[filtered_df['sentiment'] == sentiment]['confidence'])
                x = np.linspace(0, 1, 100)
                plt.plot(x, kde(x), 
                         color=COLORS[sentiment],
                         label=f'{sentiment.title()} KDE')
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.legend()
    col2.pyplot(plt)
    plt.close()
    
    tab.subheader("Confidence Distribution by Sentiment")
    plt.figure(figsize=(8, 4))
    sns.boxplot(
        x='sentiment',
        y='confidence',
        data=filtered_df,
        palette=COLORS
    )
    plt.title("Confidence Distribution per Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Confidence")
    tab.pyplot(plt)
    plt.close()
    
    tab.subheader("Sentiment Trend Analysis")
    trend_df = filtered_df.copy()
    trend_df['sentiment_score'] = trend_df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
    trend_df['row_index'] = range(len(trend_df))
    window_size = max(3, len(trend_df) // 20)
    trend_df['sentiment_ma'] = trend_df['sentiment_score'].rolling(window=window_size, center=True).mean()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    ax1.scatter(
        trend_df['row_index'],
        trend_df['sentiment_score'],
        color=trend_df['sentiment'].map(lambda x: COLORS.get(x.lower(), "#999999")),
        alpha=0.6,
        s=30,
        label='Sentiment Score'
    )
    valid_ma = trend_df[~trend_df['sentiment_ma'].isna()]
    if len(valid_ma) > 1:
        ax1.plot(
            valid_ma['row_index'],
            valid_ma['sentiment_ma'],
            color=COLORS['primary'],
            linewidth=2,
            label='Sentiment Trend (MA)'
        )
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel("Sentiment Score", color=COLORS['primary'], fontsize=11)
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax2 = ax1.twinx()
    ax2.plot(
        trend_df['row_index'],
        trend_df['confidence'],
        color=COLORS['accent'],
        linestyle='-',
        marker='.',
        markersize=3,
        alpha=0.7,
        label='Confidence'
    )
    ax2.set_ylabel("Confidence", color=COLORS['accent'], fontsize=11)
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax2.set_ylim(0, 1.05)
    plt.title("Sentiment and Confidence Trend Analysis", fontsize=14, pad=15)
    plt.xlabel("Review Sequence", fontsize=11)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.95)
    ax1.grid(True, linestyle='--', alpha=0.2)
    fig.tight_layout()
    tab.pyplot(fig)
    plt.close()

def main():
    st.set_page_config(page_title="Advanced Sentiment Analysis", page_icon="📊", layout="wide")
    st.sidebar.title("Sentiment Analysis Dashboard")
    st.sidebar.markdown("Analyze sentiments, aspects, or batch data with advanced visualizations.")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.1)
    
    @st.cache_resource
    def load_cached_models():
        with st.spinner("Loading models..."):
            return load_model_and_dependencies()
            
    model_dependencies = load_cached_models()
    
    plt.clf()
    
    tab1, tab2, tab3, tab4 = st.tabs(["General Sentiment Analysis", "Aspect-Based Analysis", "Batch Processing", "Sentiment Comparison"])
    
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
                # Attach vader_scores to filtered_df for downloadable plots
                filtered_df = pd.DataFrame([{"Sentiment": label, "Confidence": confidence}])
                filtered_df.vader_scores = vader_scores
                
                # Display Overall Sentiment
                tab1.subheader("Overall Sentiment")
                sentiment_emoji = get_sentiment_emoji(label)
                tab1.metric(
                    label="Predicted Sentiment",
                    value=f"{label.title()} {sentiment_emoji}",
                    delta=f"Confidence: {confidence:.2%}"
                )
                
                # Generate Word Cloud
                if not word_sentiments_df.empty:
                    tab1.subheader("Word Cloud Visualization")
                    sentiment_weights = word_sentiments_df.set_index("Word")["Compound"].to_dict()
                    
                    def sentiment_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                        sentiment_score = sentiment_weights.get(word, 0)
                        if sentiment_score > 0:
                            return COLORS["positive"]
                        elif sentiment_score < 0:
                            return COLORS["negative"]
                        else:
                            return COLORS["neutral"]
                    
                    # Create word cloud text based on word frequency
                    word_freq = word_sentiments_df["Word"].value_counts().to_dict()
                    wordcloud_text = " ".join([word + " " * min(int(freq * 10), 50) for word, freq in word_freq.items()])
                    
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        color_func=sentiment_color_func,
                        prefer_horizontal=0.9,
                        max_words=50,
                        contour_width=1,
                        contour_color='gray',
                        collocations=False
                    ).generate(wordcloud_text)
                    
                    plt.figure(figsize=(6, 3))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.title("Word Cloud (Sentiment-Based Coloring)", pad=15)
                    tab1.pyplot(plt)
                    plt.close()
                
                generate_sentiment_visualizations(label, confidence, probs, vader_scores, word_sentiments_df, tab1)
                
                # Add visualization options for downloadable plots
                tab1.subheader("Visualization Options")
                chart_options = tab1.multiselect(
                    "Select which visualizations to show:",
                    options=["Sentiment Distribution", "Word Sentiment Heatmap", "VADER Sentiment Line Chart", "VADER Sentiment Pie Chart", "Word Cloud"],
                    default=["Sentiment Distribution", "Word Sentiment Heatmap", "VADER Sentiment Line Chart", "VADER Sentiment Pie Chart", "Word Cloud"],
                    key="chart_options_tab1"
                )
                
                if chart_options:
                    tab1.subheader("Downloadable Visualizations")
                    create_downloadable_plots(
                        filtered_df, 
                        word_sentiments_df, 
                        tab1, 
                        "general", 
                        chart_options
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
    
    with tab4:
        tab4.header("⚖️ Sentiment Comparison")
        tab4.markdown("Enter multiple texts to compare their sentiments.")
        
        texts = []
        for i in range(3):
            text = tab4.text_area(f"Text {i+1}:", height=100, key=f"compare_text_{i}")
            if text.strip():
                texts.append((f"Text {i+1}", text))
        
        if len(texts) > 1 and tab4.button("Compare Sentiments", key="compare_sentiments"):
            with st.spinner("Comparing sentiments..."):
                comparison_data = []
                for name, text in texts:
                    label, confidence, probs = classify_sentence(text, *model_dependencies)
                    analyzer = SentimentIntensityAnalyzer()
                    vader_scores = analyzer.polarity_scores(text)
                    comparison_data.append({
                        "Text": name,
                        "Sentiment": label,
                        "Confidence": confidence,
                        "Positive": probs[2],
                        "Neutral": probs[1],
                        "Negative": probs[0],
                        "VADER Compound": vader_scores['compound']
                    })
                comparison_df = pd.DataFrame(comparison_data)
                
                tab4.subheader("Comparison Results")
                tab4.dataframe(comparison_df)
                
                plt.figure(figsize=(8, 4))
                comparison_melted = comparison_df.melt(
                    id_vars=["Text"],
                    value_vars=["Positive", "Neutral", "Negative"],
                    var_name="Sentiment",
                    value_name="Probability"
                )
                sns.barplot(
                    x="Text",
                    y="Probability",
                    hue="Sentiment",
                    data=comparison_melted,
                    palette=COLORS
                )
                plt.title("Sentiment Probabilities Comparison")
                plt.xlabel("Text")
                plt.ylabel("Probability")
                plt.legend(title="Sentiment")
                tab4.pyplot(plt)
                plt.close()
                
                plt.figure(figsize=(8, 4))
                sns.lineplot(
                    x="Text",
                    y="VADER Compound",
                    data=comparison_df,
                    marker="o",
                    color=COLORS["primary"]
                )
                plt.title("VADER Compound Score Comparison")
                plt.xlabel("Text")
                plt.ylabel("VADER Compound")
                tab4.pyplot(plt)
                plt.close()
                
                csv = comparison_df.to_csv(index=False)
                tab4.download_button(
                    label="Download Comparison as CSV",
                    data=csv,
                    file_name="sentiment_comparison.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()