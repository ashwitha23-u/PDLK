import nltk
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize

# ----------------------------- NLTK Setup -----------------------------
# Set up nltk_data path and only download if not already present
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

# --------------------------- Helper Functions --------------------------
def read_sentences(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sent_tokenize(f.read())
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []

def jaccard_similarity(str1, str2):
    words1 = set(word_tokenize(str1.lower()))
    words2 = set(word_tokenize(str2.lower()))
    return len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0

def cosine_tfidf_similarity(s1, s2, vectorizer):
    vectors = vectorizer.transform([s1, s2])
    return cosine_similarity(vectors)[0][1]

# ------------------------- Main Plagiarism Function -------------------------
def detect_plagiarism_with_heatmap(file1, file2):
    sentences1 = read_sentences(file1)
    sentences2 = read_sentences(file2)

    if not sentences1 or not sentences2:
        print("⚠️ Please provide valid files.")
        return

    all_sentences = sentences1 + sentences2
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    print("\n🔍 Starting Sentence-by-Sentence Comparison:\n")
    plagiarism_results = []

    for i, s1 in enumerate(sentences1):
        for j, s2 in enumerate(sentences2):
            jaccard = jaccard_similarity(s1, s2)
            cosine_sim = cosine_tfidf_similarity(s1, s2, vectorizer)

            print(f"🟡 Comparing:")
            print(f"[File1 - Sentence {i+1}]: {s1}")
            print(f"[File2 - Sentence {j+1}]: {s2}")
            print(f"🔹 Jaccard Similarity: {jaccard:.2f}")
            print(f"🔹 Cosine Similarity : {cosine_sim:.2f}")
            print("-" * 60)

            if jaccard > 0.2 or cosine_sim > 0.5:
                plagiarism_results.append((i+1, s1, j+1, s2, jaccard, cosine_sim))

    print("\n✅ Detected Plagiarism Cases:")
    if not plagiarism_results:
        print("No significant plagiarism detected.")
    else:
        for res in plagiarism_results:
            print(f"\n📝 File1 - Sentence {res[0]}: {res[1]}")
            print(f"📄 File2 - Sentence {res[2]}: {res[3]}")
            print(f"🔸 Jaccard Similarity: {res[4]:.2f}")
            print(f"🔸 Cosine Similarity : {res[5]:.2f}")
            print("=" * 70)

    # ------------------------- Heatmap Visualization -------------------------
    print("\n📈 Generating Similarity Heatmap...")
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    labels = [f"F1-S{i+1}" for i in range(len(sentences1))] + [f"F2-S{j+1}" for j in range(len(sentences2))]
    df_sim = pd.DataFrame(sim_matrix, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_sim, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={'label': 'Cosine Similarity'})
    plt.title("🔍 Sentence Similarity Heatmap (TF-IDF + Cosine)", fontsize=14)
    plt.xlabel("Sentences")
    plt.ylabel("Sentences")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ---------------------------- Run Script -----------------------------
file1 = input("📄 Enter path of the first document: ")
file2 = input("📄 Enter path of the second document: ")

detect_plagiarism_with_heatmap(file1, file2)
