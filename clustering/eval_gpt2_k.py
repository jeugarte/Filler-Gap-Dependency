import sys
from matplotlib import pyplot as plt
import numpy as np
import torch, random
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.decomposition import PCA, TruncatedSVD
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

    X_embedded = TruncatedSVD(n_components=3, n_iter=5).fit_transform(hidden_states_np)

    print(f"Length after TSNE: {len(X_embedded)}")
    print(f"Length of a state after TSNE: {len(X_embedded[0])}")
    
    optimal_n = silhouette(embedding=X_embedded)
    print(f"Optimal Number of Clusters: {optimal_n}")
    
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

    # plt.figure(figsize=(10, 8)) 
    # for i in range(len(X_embedded)):
    #     word = word_labels[i]
    #     color = colors.get(word) 
    #     plt.scatter(X_embedded[i, 0], X_embedded[i, 1], color=color, label=word)
    #     plt.annotate(word,
    #                 (X_embedded[i, 0], X_embedded[i, 1]),
    #                 textcoords="offset points",
    #                 xytext=(5,2),
    #                 ha='center',
    #                 fontsize=8)

    # plt.xlabel('t-SNE Feature 1')
    # plt.ylabel('t-SNE Feature 2')
    # plt.title('t-SNE Visualization with Word Labels and Colors')
    # plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    word_labels_array = np.array(word_labels)

    for category in colors:
        indices = word_labels_array == category
        ax.scatter(X_embedded[indices, 0], X_embedded[indices, 1], X_embedded[indices, 2], 
                c=colors[category], label=category)

    ax.legend() 
    plt.show()
    



def k_means_pca(hidden_file):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    sentences = preprocess(hidden_file=hidden_file)
    hidden_states = extract_hidden(sentences=sentences, tokenizer=tokenizer, model=model)

    print(f"Length before TSNE: {len(hidden_states)}")
    print(f"Length of a state before TSNE: {len(hidden_states[0])}")

    hidden_states_np = np.array(hidden_states)

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(hidden_states_np)
    k = KMeans(n_clusters=4, random_state=42).fit(reduced_data)
    labels = k.labels_

    optimal = silhouette(embedding=reduced_data)

    # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.title('Hidden States Clustered via K-Means')
    # plt.show()

    word_labels = ['THAT'] * 20 + ['WHETHER'] * 20 + ['WHY'] * 20 + ['HOW'] * 20 + ['WHEN'] * 20 +  ['WHERE'] * 20
    colors = {
        'THAT': 'red',
        'WHETHER': 'blue',
        'WHY': 'green',
        'HOW': 'purple',
        'WHEN': 'orange',
        'WHERE': 'brown'
    }

    # plt.figure(figsize=(10, 8)) 
    # for i in range(len(reduced_data)):
    #     word = word_labels[i]
    #     color = colors.get(word) 
    #     plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color=color, label=word)
    #     plt.annotate(word,
    #                 (reduced_data[i, 0], reduced_data[i, 1]),
    #                 textcoords="offset points",
    #                 xytext=(5,2),
    #                 ha='center',
    #                 fontsize=8)

    # plt.xlabel('t-SNE Feature 1')
    # plt.ylabel('t-SNE Feature 2')
    # plt.title('t-SNE Visualization with Word Labels and Colors')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    word_labels_array = np.array(word_labels)

    for category in colors:
        indices = word_labels_array == category
        ax.scatter(reduced_data[indices, 0], reduced_data[indices, 1], reduced_data[indices, 2], 
                c=colors[category], label=category)

    ax.legend()
    plt.show()



if __name__ == "__main__":
    hidden_file = sys.argv[1]
    tsne_or_pca = sys.argv[2]
    if tsne_or_pca == "tsne":
        k_means(hidden_file)
    else:
        k_means_pca(hidden_file)