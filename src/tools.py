import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans
import hdbscan
from sklearn.cluster import AgglomerativeClustering


# Function to generate embeddings
def generate_embeddings(texts, tokenizer, model, batch_size=16):
    """
    Generate embeddings for a list of texts using DistilRoBERTa with mean pooling.
    """
    embeddings = []
    # Wrap the loop with tqdm for a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings", unit="batch"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling for sentence representation
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

            # Compute mean of embeddings, taking into account the attention mask
            masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # Shape: (batch_size, seq_length, hidden_size)
            sum_embeddings = masked_embeddings.sum(dim=1)  # Shape: (batch_size, hidden_size)
            count_embeddings = attention_mask.sum(dim=1)  # Shape: (batch_size)

            # Avoid division by zero and compute mean
            mean_embeddings = sum_embeddings / count_embeddings.unsqueeze(-1).clamp(min=1e-9)  # Shape: (batch_size, hidden_size)
            embeddings.append(mean_embeddings)
    return torch.cat(embeddings, dim=0)

def reduce_dimensionality(embeddings, n_components=300, algo='none'):
    """
    Reduce the dimensionality of embeddings using PCA.
    """
    if algo == 'pca':
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    elif algo == 'umap':
        reducer = umap.UMAP(n_components=n_components)
        reduced_embeddings = reducer.fit_transform(embeddings.cpu().numpy())
    elif algo == 'none':
        reduced_embeddings = embeddings.cpu().numpy()
    return reduced_embeddings

def perform_clustering(embeddings, n_clusters=5, algo='kmeans'):
    """
    Cluster embeddings using KMeans.
    """
    reduced_embeddings = reduce_dimensionality(embeddings)
    if algo == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_embeddings)
    elif algo == 'hdbscan':
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
        clusters = clusterer.fit_predict(reduced_embeddings)
    elif algo == 'agg':
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = agg_clustering.fit_predict(reduced_embeddings)
    return clusters