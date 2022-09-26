import logging
import os
import gensim
import smart_open
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward
from scipy.spatial import distance
from scipy.spatial.distance import pdist


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Set file names for train and test data
test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')
lee_test_file = os.path.join(test_data_dir, 'lee.cor')

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
test_texts = list(read_corpus(lee_train_file, tokens_only=True))

# print(train_corpus[:2])
# print(test_corpus[:2])

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=100)
model.build_vocab(train_corpus)

# print(f"Word 'penalty' appeared {model.wv.get_vecattr('penalty', 'count')} times in the training corpus.")

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)



# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)

#     second_ranks.append(sims[1])

# counter = collections.Counter(ranks)
# print(counter)

# print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

# # Pick a random document from the corpus and infer a vector from the model
# doc_id = random.randint(0, len(train_corpus) - 1)

# # Compare and print the second-most-similar document
# print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# sim_id = second_ranks[doc_id]
# print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))






X = np.array([model.infer_vector(x) for x in test_texts])


plt.figure(figsize=(10, 7))
plt.title("Dendogram with line")
clusters = linkage(X,
            method='ward', 
            metric="euclidean")
dendrogram(clusters)
plt.axhline(y = 125, color = 'r', linestyle = '-')
plt.show()

# Get number of clusters


clustering_model = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
clustering_model.fit(X)
clustering_model.labels_


pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
pc1_values = pcs[:,0]
pc2_values = pcs[:,1]
sns.scatterplot(x=pc1_values, y=pc2_values)
plt.show()


clustering_model_pca = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
clustering_model_pca.fit(pcs)


data_labels_pca = clustering_model_pca.labels_


sns.scatterplot(x=pc1_values, 
                y=pc2_values,
                hue=data_labels_pca,
                palette="rainbow")

plt.show()



















clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(clustering, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()











y = pdist(X)

Z = ward(y)

fcluster(Z, 0.9, criterion='distance')

[x[:5] for x in test_texts]

linkage_matrix = ward(y) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(Z, orientation="right", labels=[' '.join(x[:10]) for x in test_texts]);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.savefig('/Users/huwwaters/Desktop/clustering.png', dpi=600) #save figure as ward_clusters

plt.show()





X = np.array([model.infer_vector(x) for x in test_texts])

plt.figure(figsize=(10, 7))
plt.title("Dendrogram")

clusters = linkage(X, method='ward', metric="euclidean")

dendrogram(Z=clusters, orientation="right", labels=[' '.join(x[:10]) for x in test_texts])

plt.tick_params(
    axis= 'x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off'
    )

# plt.savefig('/Users/huwwaters/Desktop/clustering.png', dpi=600) #save figure as ward_clusters
plt.show()











df = pd.DataFrame({'Text': test_texts, 'Cluster': clustering_model.labels_})

from sklearn.feature_extraction.text import TfidfVectorizer

td_idf_text = df[df['Cluster']==0]['Text'].tolist()
td_idf_text = [' '.join(x) for x in td_idf_text]

vectorizer = TfidfVectorizer(ngram_range=(1,5), stop_words='english')
X_tfidf = vectorizer.fit_transform(td_idf_text)
vectorizer.get_feature_names_out()

feature_array = np.array(vectorizer.get_feature_names())
tfidf_sorting = np.argsort(X_tfidf.toarray()).flatten()[::-1]

n = 50
top_n = feature_array[tfidf_sorting][:n]
top_n