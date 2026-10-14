
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datasets import load_dataset
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import (
    CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
)

# ---------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------
st.set_page_config(page_title="AG News | Text Visualization",
                   page_icon="📰", layout="wide")

SEED = 42
PER_CATEGORY = 500
N_TOPICS = 4
CATEGORIES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
CATEGORY_ORDER = list(CATEGORIES.values())
CATEGORY_COLORS = {
    "World": "#6366F1", "Sports": "#10B981",
    "Business": "#F59E0B", "Sci/Tech": "#EC4899"
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | {
    "reuters", "ap", "afp", "said", "says", "quot"
})
TOKEN_PATTERN = r"(?u)\b[a-z]{3,}\b"


def clean_text(value):
    """Decode entities before stripping HTML and punctuation."""
    value = html.unescape(str(value)).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"<[^>]*>", " ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@st.cache_data(show_spinner="Loading AG News data...")
def load_data():
    # Prefer a bundled file when deployed, otherwise use Hugging Face.
    paths = [
        Path("data/train-00000-of-00001.parquet"),
        Path("train-00000-of-00001.parquet"),
        Path("data/ag_news.csv"),
    ]
    local = next((p for p in paths if p.exists()), None)
    if local is not None:
        source = pd.read_parquet(local) if local.suffix == ".parquet" else pd.read_csv(local)
        source_description = f"Local file: {local}"
    else:
        source = load_dataset("fancyzhx/ag_news", split="train").to_pandas()
        source_description = "Hugging Face: fancyzhx/ag_news (train)"

    if not {"text", "label"}.issubset(source.columns):
        raise ValueError("Dataset must have 'text' and 'label' columns.")
    source = source[["text", "label"]].dropna().copy()
    source["text"] = source["text"].astype(str)
    source = source[source["text"].str.strip().ne("")]
    source["label"] = pd.to_numeric(source["label"], errors="coerce")
    source = source[source["label"].isin(CATEGORIES)].copy()
    if any((source["label"] == i).sum() < PER_CATEGORY for i in CATEGORIES):
        raise ValueError("Each AG News category must have at least 500 valid articles.")

    sample = (source.groupby("label", group_keys=False)
              .sample(n=PER_CATEGORY, random_state=SEED)
              .reset_index(drop=True))
    sample["label"] = sample["label"].astype(int)
    sample["category"] = sample["label"].map(CATEGORIES)
    sample["clean_text"] = sample["text"].map(clean_text)
    sample["word_count"] = sample["clean_text"].str.split().str.len()
    sample = sample[sample["clean_text"].ne("")].reset_index(drop=True)
    return sample, source_description


@st.cache_resource(show_spinner="Training LDA and preparing document projection...")
def fit_models(texts):
    count_vectorizer = CountVectorizer(
        stop_words=STOP_WORDS, max_features=2500,
        min_df=5, max_df=0.85, token_pattern=TOKEN_PATTERN
    )
    count_matrix = count_vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, random_state=SEED,
        learning_method="batch", max_iter=15, n_jobs=1
    )
    document_topics = lda.fit_transform(count_matrix)

    tfidf_vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS, max_features=2500,
        min_df=2, max_df=0.95, token_pattern=TOKEN_PATTERN
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=2, random_state=SEED)
    coordinates = svd.fit_transform(tfidf_matrix)
    return (lda, document_topics, count_vectorizer.get_feature_names_out(),
            coordinates, svd.explained_variance_ratio_)


@st.cache_data(show_spinner=False)
def word_statistics(texts):
    """Use identical vocabulary and preprocessing for both rankings."""
    count_vectorizer = CountVectorizer(
        stop_words=STOP_WORDS, min_df=2, max_df=0.95,
        token_pattern=TOKEN_PATTERN
    )
    counts_matrix = count_vectorizer.fit_transform(texts)
    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=count_vectorizer.vocabulary_,
        stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN,
        norm="l2"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    names = count_vectorizer.get_feature_names_out()
    result = pd.DataFrame({
        "Word": names,
        "Count": np.asarray(counts_matrix.sum(axis=0)).ravel().astype(int),
        "Average TF-IDF": np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    })
    return result


def horizontal_bars(frame, value, title, palette="Blues", percent=False):
    chart = px.bar(
        frame.sort_values(value), x=value, y="Word", orientation="h",
        color=value, color_continuous_scale=palette, title=title,
        hover_data={"Word": True, value: True}
    )
    chart.update_layout(
        template="plotly_white", coloraxis_showscale=False,
        height=max(400, 22 * len(frame) + 160),
        margin=dict(l=25, r=25, t=65, b=30)
    )
    if percent:
        chart.update_xaxes(tickformat=".1%")
    st.plotly_chart(chart, use_container_width=True)


def interpretation_and_limits(interpretation, limitations):
    st.subheader("Interpretation")
    st.info(interpretation)
    st.subheader("Limitations")
    st.warning(limitations)


# ---------------------------------------------------------
# 2. Dataset and controls
# ---------------------------------------------------------
st.title("📰 AG News – Interactive Text Explorer")
st.markdown("**DATS 6401 – Visualization of Complex Data**")
st.markdown("**Week 7 Assignment: Text Data Visualization**")
st.write(
    "Explore word frequencies, latent topics, two-dimensional document "
    "representations, and differences between raw word counts and TF-IDF."
)
st.divider()

try:
    df, source_description = load_data()
    texts = tuple(df["clean_text"].tolist())
    lda, document_topics, vocabulary, coordinates, explained_variance = fit_models(texts)
except Exception as exc:
    st.error("The dataset or model could not be prepared.")
    st.exception(exc)
    st.stop()

topic_names = [f"Topic {i+1}" for i in range(N_TOPICS)]
dominant_ids = document_topics.argmax(axis=1)
df["dominant_topic"] = [topic_names[i] for i in dominant_ids]

st.sidebar.title("Visualization Controls")
selected_category = st.sidebar.selectbox(
    "News Category", ["All Categories"] + CATEGORY_ORDER
)
top_n = st.sidebar.slider("Number of Top Words", 5, 30, 15, 5)
st.sidebar.divider()
st.sidebar.write(f"**Dataset:** AG News")
st.sidebar.write(f"**Sample:** {len(df):,} articles")
st.sidebar.write(f"**Categories:** {df['category'].nunique()}")
st.sidebar.write("**Topic Model:** LDA (4 topics)")
st.sidebar.write("**Projection:** TF-IDF + TruncatedSVD")
st.sidebar.write(f"**Random State:** {SEED}")

filtered_df = (
    df if selected_category == "All Categories"
    else df[df["category"].eq(selected_category)]
)
filtered_texts = tuple(filtered_df["clean_text"].tolist())

# ---------------------------------------------------------
# 3. Overview
# ---------------------------------------------------------
st.header("1. Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", f"{len(df):,}")
c2.metric("Categories", df["category"].nunique())
c3.metric("Average Cleaned Words", f"{df['word_count'].mean():.1f}")
c4.metric("Selected Documents", f"{len(filtered_df):,}")

counts = (df["category"].value_counts().reindex(CATEGORY_ORDER)
          .rename_axis("Category").reset_index(name="Documents"))
fig = px.bar(
    counts, x="Category", y="Documents", color="Category",
    color_discrete_map=CATEGORY_COLORS,
    title="AG News Category Distribution"
)
fig.update_layout(template="plotly_white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Explore Original News Articles"):
    st.dataframe(
        filtered_df[["category", "text", "word_count"]],
        hide_index=True, use_container_width=True
    )
st.caption(
    f"Data source: {source_description}. A reproducible stratified sample "
    "was drawn with 500 articles per category before preprocessing. "
    "It does not represent the source dataset's natural category proportions."
)
st.divider()

# ---------------------------------------------------------
# 4. Word frequencies
# ---------------------------------------------------------
st.header("2. Word Frequency Visualization")
st.write("Most frequent words after cleaning and stop-word removal.")
try:
    stats = word_statistics(filtered_texts)
except ValueError:
    st.error("There are too few usable words for this category.")
    st.stop()
raw_top = stats.nlargest(top_n, "Count")
horizontal_bars(raw_top, "Count", f"Top {top_n} Words – {selected_category}")
with st.expander("Word Frequency Table"):
    st.dataframe(
        stats.sort_values("Count", ascending=False).head(50)[["Word", "Count"]],
        hide_index=True, use_container_width=True
    )
most_common = raw_top.iloc[0]
interpretation_and_limits(
    f"The most frequent term in this selection is '{most_common['Word']}' "
    f"({int(most_common['Count'])} occurrences). These counts show recurring "
    "vocabulary, not necessarily the most informative terms.",
    "Frequency ignores context and meaning. Long documents can contribute "
    "more occurrences, and tokenization, stop words, and frequency cutoffs "
    "affect which words appear."
)
st.divider()

# ---------------------------------------------------------
# 5. LDA
# ---------------------------------------------------------
st.header("3. Topic Modeling and Visualization")
st.write(
    "LDA finds recurring patterns of word co-occurrence without using "
    "the original AG News category labels during training."
)
selected_topic = st.selectbox("Select a Topic to Explore", topic_names)
topic_id = topic_names.index(selected_topic)
weights = lda.components_[topic_id]
indices = np.argsort(weights)[-12:][::-1]
topic_words = pd.DataFrame({
    "Word": vocabulary[indices],
    "Probability": (weights / weights.sum())[indices]
})
horizontal_bars(
    topic_words, "Probability",
    f"Top Words in {selected_topic}", "Viridis", percent=True
)
st.caption("Bar values are estimated P(word | topic), not article probabilities.")

st.subheader("Topic Prevalence")
prevalence = pd.DataFrame({
    "Topic": topic_names,
    "Average Proportion": document_topics.mean(axis=0)
})
fig = px.bar(
    prevalence, x="Topic", y="Average Proportion", color="Topic",
    title="Average Topic Proportion Across All Documents"
)
fig.update_layout(template="plotly_white", showlegend=False,
                  yaxis_tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Representative News Articles")
indices = np.argsort(document_topics[:, topic_id])[-5:][::-1]
representatives = df.iloc[indices][["category", "text"]].copy()
representatives["Topic Probability"] = document_topics[indices, topic_id]
st.dataframe(representatives, hide_index=True, use_container_width=True)

st.subheader("Topics Across News Categories")
heatmap_input = pd.DataFrame({
    "Category": df["category"], "Topic": df["dominant_topic"]
})
heatmap = pd.crosstab(
    heatmap_input["Category"], heatmap_input["Topic"], normalize="index"
).reindex(index=CATEGORY_ORDER, columns=topic_names, fill_value=0)
fig = px.imshow(
    heatmap, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
    title="Dominant Topic Distribution by Category",
    labels={"x": "LDA Topic", "y": "News Category", "color": "Share"}
)
fig.update_layout(template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Each row sums to approximately 100% (rounding may apply). "
    "The heatmap assigns each article to its highest-probability topic; "
    "LDA itself allows mixtures of topics."
)
interpretation_and_limits(
    f"{selected_topic} has prominent words including "
    f"{', '.join(topic_words['Word'].head(5))}. "
    "Inspect the representative articles to decide how to describe this "
    "statistical topic; topic numbers are not predefined news categories.",
    "Topics depend on the sample, preprocessing, number of topics, and "
    "random initialization. Dominant-topic labels discard uncertainty "
    "and should not be treated as ground-truth classifications."
)
st.divider()

# ---------------------------------------------------------
# 6. 2D document projection
# ---------------------------------------------------------
st.header("4. Document Embedding Projection")
st.write(
    "TF-IDF document vectors are reduced to two dimensions using "
    "TruncatedSVD (a linear lexical projection, not a neural embedding)."
)
projection = pd.DataFrame({
    "Component 1": coordinates[:, 0],
    "Component 2": coordinates[:, 1],
    "Category": df["category"].values,
    "Dominant Topic": df["dominant_topic"].values,
    "Article": df["text"].str.slice(0, 180).values
})
color_by = st.radio(
    "Color Documents By", ["Category", "Dominant Topic"], horizontal=True
)
fig = px.scatter(
    projection, x="Component 1", y="Component 2", color=color_by,
    hover_data=["Article", "Category", "Dominant Topic"],
    opacity=0.65, title="2D Projection of AG News Documents",
    color_discrete_map=CATEGORY_COLORS if color_by == "Category" else None
)
fig.update_traces(marker=dict(size=6))
fig.update_layout(template="plotly_white", height=620)
st.plotly_chart(fig, use_container_width=True)
variance = float(explained_variance.sum())
st.metric("Explained Variance (2 Components)", f"{variance:.2%}")
if variance < 0.10:
    st.warning(
        "These two components retain less than 10% of the variation "
        "measured by TruncatedSVD. This is a very lossy projection; "
        "do not interpret apparent clusters as strong evidence of "
        "semantic similarity."
    )
interpretation_and_limits(
    "Each point represents an article. Changing the color reveals how "
    "original categories and dominant LDA topics overlap in the same "
    "two-dimensional lexical representation.",
    f"The two displayed components explain only {variance:.2%} of "
    "the measured variation. Two-dimensional distances can be misleading; "
    "TF-IDF ignores word order and semantic context. "
    "No claim of accurate semantic clustering is made."
)
st.divider()

# ---------------------------------------------------------
# 7. Raw counts vs TF-IDF
# ---------------------------------------------------------
st.header("5. TF-IDF vs Raw Word Counts")
st.write(
    "Compare two ranking methods on the SAME selected articles and "
    "the SAME vocabulary. Their numeric scales are different."
)
method = st.radio(
    "Choose Word Ranking Method", ["Raw Counts", "Average TF-IDF"],
    horizontal=True
)
column = "Count" if method == "Raw Counts" else "Average TF-IDF"
ranked = stats.nlargest(top_n, column)
horizontal_bars(
    ranked, column, f"{method} – {selected_category}", "Teal"
)
with st.expander("Compare Both Rankings Side by Side"):
    comparison = stats.copy()
    comparison["Raw Rank"] = comparison["Count"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["TF-IDF Rank"] = comparison["Average TF-IDF"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["Rank Difference"] = (
        comparison["Raw Rank"] - comparison["TF-IDF Rank"]
    )
    st.dataframe(
        comparison.sort_values("Raw Rank").head(50),
        hide_index=True, use_container_width=True
    )
raw_words = set(stats.nlargest(top_n, "Count")["Word"])
tfidf_words = set(stats.nlargest(top_n, "Average TF-IDF")["Word"])
st.write(
    f"**Top-{top_n} overlap:** {len(raw_words & tfidf_words)} of {top_n} "
    "words appear in both rankings."
)
interpretation_and_limits(
    "Raw counts measure total token occurrences; mean TF-IDF scores "
    "measure average normalized word weight across the selected documents. "
    "The rank comparison shows how the ordering changes.",
    "TF-IDF is not a semantic importance score. Its mean is influenced by "
    "document length, term rarity, vocabulary selection, and corpus scope. "
    "Do not compare the raw numerical heights across the two methods."
)
st.divider()

# ---------------------------------------------------------
# 8. Methodology
# ---------------------------------------------------------
st.header("6. Methodology and Reproducibility")
with st.expander("Dataset and Sampling"):
    st.markdown(
        f"**Source:** {source_description}\n\n"
        f"**Sample:** {len(df):,} articles; up to {PER_CATEGORY} per category.\n\n"
        f"**Random seed:** {SEED}.\n\n"
        "The sample is stratified and is not representative of natural "
        "category proportions."
    )
with st.expander("Text Preprocessing"):
    st.markdown(
        "HTML entities are decoded, text is lowercased, URLs and markup "
        "are removed, and only alphabetic tokens of length 3+ remain. "
        "English stop words and selected source terms are excluded during "
        "vectorization. No stemming or lemmatization is used."
    )
with st.expander("LDA Model Configuration"):
    st.markdown(
        f"**Topics:** {N_TOPICS}; **maximum vocabulary:** 2,500; "
        "**min_df:** 5; **max_df:** 0.85; **iterations:** 15; "
        f"**learning method:** batch; **random state:** {SEED}. "
        "Four topics are an exploratory modeling choice, not a "
        "claim that topics equal news categories."
    )
with st.expander("Projection Method"):
    st.markdown(
        "**Representation:** L2-normalized TF-IDF; "
        "**maximum vocabulary:** 2,500; **min_df:** 2; "
        "**max_df:** 0.95; **reduction:** TruncatedSVD; "
        f"**components:** 2; **random state:** {SEED}. "
        "This is a lexical linear projection, not a deep-learning embedding."
    )
with st.expander("Overall Limitations"):
    st.markdown(
        "1. Balanced sample may not represent the original distribution.\n"
        "2. Frequency and TF-IDF do not capture full semantic meaning.\n"
        "3. LDA topics require subjective interpretation.\n"
        "4. Topic count and preprocessing affect results.\n"
        "5. Two-dimensional SVD discards substantial information.\n"
        "6. News data and model results may not generalize."
    )
st.markdown(
    "**Dataset:** [AG News on Hugging Face]"
    "(https://huggingface.co/datasets/fancyzhx/ag_news)"
)
st.caption("DATS 6401 – Week 7 | Text Data Visualization")
