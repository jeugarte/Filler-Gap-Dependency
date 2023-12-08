import sys
from matplotlib import pyplot as plt
import numpy as np
import torch, random
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE


def preprocess(hidden_file):
    with open(hidden_file, 'r') as sentence_file:
        sentences = sentence_file.read().splitlines()
        return sentences
    
def extract_hidden(sentences, tokenizer, model):
    hidden_states = []

    for sentence in sentences:
        inputs_tokens = tokenizer(sentence, return_tensors="pt")
        output_tokens = model(**inputs_tokens)
        curr_hidden_states = output_tokens.last_hidden_state

        for word_index in range(curr_hidden_states.size(1)):
            word_hidden_state = curr_hidden_states[0, word_index, :].tolist()
            input_word = tokenizer.decode(inputs_tokens['input_ids'][0, word_index])
            if input_word.strip() in ['that', 'where', 'when', 'how', 'why', 'whether']:
                hidden_states.append(word_hidden_state)

    return hidden_states



def silhouette(embedding):
    silhouette_scores = []
    cluster_range = list(range(2, 10))

    for n_clusters in cluster_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(embedding)
        silhouette_avg = silhouette_score(embedding, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is:", silhouette_avg)

    # plt.plot(cluster_range, silhouette_scores)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Scores for Various Numbers of Clusters')
    # plt.show()
    
    return cluster_range[np.argmax(silhouette_scores)]



def k_means(hidden_file):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    sentences = preprocess(hidden_file=hidden_file)
    hidden_states = extract_hidden(sentences=sentences, tokenizer=tokenizer, model=model)

    print(f"Length before TSNE: {len(hidden_states)}")
    print(f"Length of a state before TSNE: {len(hidden_states[0])}")

    hidden_states_np = np.array(hidden_states)

    X_embedded = TSNE(n_components=2, random_state=1).fit_transform(hidden_states_np)

    print(f"Length after TSNE: {len(X_embedded)}")
    print(f"Length of a state after TSNE: {len(X_embedded[0])}")
    
    optimal_n = silhouette(embedding=X_embedded)
    
    k = KMeans(n_clusters=optimal_n, random_state=42)
    cluster_labels = k.fit_predict(X_embedded)

    word_labels = ['THAT'] * 20 + ['WHETHER'] * 20 + ['WHY'] * 20 + ['HOW'] * 20 + ['WHEN'] * 20 +  ['WHERE'] * 20
    colors = {
        'THAT': 'red',
        'WHETHER': 'blue',
        'WHY': 'green',
        'HOW': 'purple',
        'WHEN': 'orange',
        'WHERE': 'brown'
    }

    # Plotting
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    for i in range(len(X_embedded)):
        word = word_labels[i]
        color = colors.get(word)  # Default to gray if word not in colors
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], color=color, label=word)
        plt.annotate(word,
                    (X_embedded[i, 0], X_embedded[i, 1]),
                    textcoords="offset points",
                    xytext=(5,2),
                    ha='center',
                    fontsize=8)

    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Visualization with Word Labels and Colors')
    plt.show()

    # category_labels = ['COMP'] * 40 + ['ADJ'] * 80
    # for category, marker in [('COMP', 'o'), ('ADJ', '^')]:
    #     indices = [i for i, label in enumerate(category_labels) if label == category]
    #     plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], 
    #                 c=cluster_labels[indices], label=category, marker=marker)

    # plt.xlabel('t-SNE Feature 1')
    # plt.ylabel('t-SNE Feature 2')
    # plt.title('t-SNE Visualization of Clusters with Original Categories')
    # plt.legend()
    # plt.show()
    




if __name__ == "__main__":
    hidden_file = sys.argv[1]
    k_means(hidden_file)