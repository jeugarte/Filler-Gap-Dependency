import sys
from matplotlib import pyplot as plt
import numpy as np
import torch, random
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D



def preprocess(hidden_file):
    with open(hidden_file, 'r') as sentence_file:
        sentences = sentence_file.read().splitlines()
        return sentences
    
def extract_hidden(sentences, tokenizer, model):
    hidden_states = []

    for sentence in sentences:
        inputs_tokens = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            output_tokens = model(**inputs_tokens)
        curr_hidden_states = output_tokens.last_hidden_state

        for word_index in range(curr_hidden_states.size(1)):
            input_word = tokenizer.decode([inputs_tokens['input_ids'][0, word_index]]).strip()
            if input_word.strip() in ['that', 'where', 'when', 'how', 'why', 'whether']:
                word_hidden_state = curr_hidden_states[0, word_index, :].tolist()
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    word_labels_array = np.array(word_labels)

    # Plot each category with its color and label
    for category in colors:
        indices = word_labels_array == category
        ax.scatter(X_embedded[indices, 0], X_embedded[indices, 1], X_embedded[indices, 2], 
                c=colors[category], label=category)

    ax.legend()  # Add legend
    plt.show()

    # plt.figure(figsize=(10, 8)) 
    # for i in range(len(X_embedded)):
    #     word = word_labels[i]
    #     color = colors.get(word) 
    #     plt.scatter(X_embedded[i, 1], X_embedded[i, 2], color=color, label=word)
    #     plt.annotate(word,
    #                 (X_embedded[i, 1], X_embedded[i, 2]),
    #                 textcoords="offset points",
    #                 xytext=(5,2),
    #                 ha='center',
    #                 fontsize=8)

    # plt.xlabel('t-SNE Feature 1')
    # plt.ylabel('t-SNE Feature 2')
    # plt.title('K-Means Clustering of WH-Adjucnt vs Complementizers (t-SNE)')
    # plt.show()

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
    



def k_means_pca(hidden_file):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    sentences = preprocess(hidden_file=hidden_file)
    hidden_states = extract_hidden(sentences=sentences, tokenizer=tokenizer, model=model)

    print(f"Length before TSNE: {len(hidden_states)}")
    print(f"Length of a state before TSNE: {len(hidden_states[0])}")

    hidden_states_np = np.array(hidden_states)

    # cluster_labels = k.fit_predict(X_embedded).fit()

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

    # category_labels = ['COMP'] * 40 + ['ADJ'] * 80
    # for category, marker in [('COMP', 'o'), ('ADJ', '^')]:
    #     indices = [i for i, label in enumerate(category_labels) if label == category]
    #     plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], 
    #                 c=labels[indices], label=category, marker=marker)

    # plt.xlabel('PCA Feature 1')
    # plt.ylabel('PCA Feature 2')
    # plt.title('PCA Visualization of Clusters with Original Categories')
    # plt.legend()
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(reduced_data[:,0], reduced_data[:,1], reduced_data[:,2])
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

    # plt.xlabel('PCA Feature 1')
    # plt.ylabel('PCA Feature 2')
    # plt.title('K-Means Clustering of WH-Adjucnt vs Complementizers (PCA)')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    word_labels_array = np.array(word_labels)

    # Plot each category with its color and label
    for category in colors:
        indices = word_labels_array == category
        ax.scatter(reduced_data[indices, 0], reduced_data[indices, 1], reduced_data[indices, 2], 
                c=colors[category], label=category)

    ax.legend()  # Add legend
    plt.show()







    # cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']  # Add more colors if you have more clusters

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for i in range(len(reduced_data)):
    #     ax.scatter(reduced_data[i, 0], reduced_data[i, 1], reduced_data[i, 2], 
    #             c=cluster_colors[labels[i]])

    # ax.set_xlabel('PCA Component 1')
    # ax.set_ylabel('PCA Component 2')
    # ax.set_zlabel('PCA Component 3')

    # # Show plot
    # plt.show()


if __name__ == "__main__":
    hidden_file = sys.argv[1]
    tsne_or_pca = sys.argv[2]
    if tsne_or_pca == "tsne":
        k_means(hidden_file)
    else:
        k_means_pca(hidden_file)