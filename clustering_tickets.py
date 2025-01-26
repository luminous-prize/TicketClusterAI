import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import re
import json

def remove_confidential_info(dataframe, column_name):
    
    
    email_pattern = r'\S+@\S+'
    phone_pattern = r'\+\d{10,12}'
    
    # iterating over each row for subsitution
    for index, row in dataframe.iterrows():
        text = row[column_name]
        
        
        text = re.sub(email_pattern, '{MASKED_ID}', text)
        
        
        text = re.sub(phone_pattern, '{MASKED_NUMBER}', text)
        
        # Update the DataFrame with the cleaned text
        dataframe.at[index, column_name] = text
    
    return dataframe

def model_config():
    sent_emb_model = 'multi-qa-mpnet-base-dot-v1'
    sent_emb_max_seq_length = 500
    sent_embedder = SentenceTransformer(sent_emb_model, device = 'cpu')
    sent_embedder.max_seq_length = sent_emb_max_seq_length
    return sent_embedder

def get_embedding(text):
    sent_embedder = model_config()
    embeddings = sent_embedder.encode(text)
    return embeddings

def cluster_config():
    n_clusters_max = 10  
    best_silhouette_score = -1  
    best_k = 7 # default k value to consider
    
def silhouette(n_clusters_max=10, best_silhouette_score=-1, best_k=7):
    # evaluating the best possible value of k using silhouette score
    
    data = pd.read_csv('Ticket_Data.csv')
    
    #gdpr sanitization for Subject
    data['Subject'] = data['Subject'].astype(str)
    column_name = 'Subject'
    data = remove_confidential_info(data, column_name)
    
    for k in range(5, n_clusters_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init = 10)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_k = k
            
    return best_k


def check_representative(size, cluster_data, representative_index, sorted_indices, cluster_indices):
    
    temp = representative_index
    representative_resolution = cluster_data.loc[cluster_indices[representative_index], 'Resolution Description']
    
    for i in range(size):

        if i == size-1:
            representative_index = temp
            break
        
        if not representative_resolution.strip():
            representative_index = sorted_indices[0+i]
            representative_resolution = cluster_data.loc[cluster_indices[representative_index], 'Resolution Description']
            
            if not representative_resolution.strip():
                continue
            else:
                break
        else:
            break
                
                
    return representative_index


def clustering_labels(best_k):
    # clustering the tickets using the best K
    
    data = pd.read_csv('Ticket_Data.csv')
    
    #gdpr sanitization for Subject
    data['Subject'] = data['Subject'].astype(str)
    column_name = 'Subject'
    data = remove_confidential_info(data, column_name)
    
    kmeans = KMeans(n_clusters=best_k, random_state=40, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    
    return cluster_labels


    
def get_cluster_representatives(cluster_id, cluster_data, embeddings):
    
    """
    cluster representatives (description and resolution) for a specific cluster.

    Args:
        cluster_id (int): The ID of the cluster for which representatives are computed.
        cluster_data (pd.DataFrame): DataFrame containing data for the entire cluster.
        embeddings (np.ndarray): NumPy array of embeddings for all tickets.

    Returns:
        List[str]: List of representative ticket descriptions and resolutions.
    """
    
    # Filter data for the specific cluster
    cluster_data = cluster_data[cluster_data['cluster_id'] == cluster_id]
    
    # indices of tickets in the cluster
    cluster_indices = cluster_data.index
    
    # the mean embedding of the cluster
    cluster_mean_embedding = np.mean(embeddings[cluster_indices], axis=0)
    
    #  L2 distances between each ticket's embedding and the cluster mean embedding
    distances = np.linalg.norm(embeddings[cluster_indices] - cluster_mean_embedding, axis=1)
    
    # Sorting tickets by ascending distance (closer to the mean is better)
    sorted_indices = np.argsort(distances)
        
    # Select the top representative (closest to the mean)
    representative_index = sorted_indices[0]
    
    closest_vector = embeddings[cluster_indices][representative_index]
    
    
    # Calculate cosine similarity using sklearn's cosine_similarity function
    cosine_similarity_matrix = cosine_similarity([cluster_mean_embedding], [closest_vector])

    # Extract the cosine similarity value
    similarity = cosine_similarity_matrix[0, 0]
        
    representative_index = check_representative(len(cluster_data), cluster_data, representative_index, sorted_indices, cluster_indices)


    
    # Get the description and resolution of the representative ticket
    representative_case = cluster_data.loc[cluster_indices[representative_index], 'Case Number']
    representative_system = cluster_data.loc[cluster_indices[representative_index], 'Subject']
    representative_resolution = cluster_data.loc[cluster_indices[representative_index], 'Resolution Description']
    
    if not representative_resolution.strip():
        representative_resolution = 'Unresolved Ticket'
    
    if similarity < 0.70:
        return 1
    
    return [representative_case,representative_system, representative_resolution]




def representative_indexing(cluster_labels):
    
    data = pd.read_csv('Ticket_Data.csv')
    
    #gdpr sanitization for Subject
    data['Subject'] = data['Subject'].astype(str)
    column_name = 'Subject'
    data = remove_confidential_info(data, column_name)
    data['cluster_id'] = cluster_labels
    cluster_representatives = {}
    
    # Iterating over each cluster ID
    for cluster_id in sorted(data['cluster_id'].unique()):
        representatives = get_cluster_representatives(cluster_id, data, embeddings)
        if representatives == 1:
            continue
        cluster_representatives[cluster_id] = representatives
        
     # Creating new pretty format for json
    parsed_data = {}
    for cluster_id, values in cluster_representatives.items():
        cluster_name = f"Cluster {cluster_id}"
        item_dict = {
            "cluster_id": cluster_name,
            "Ticket_Number": values[0],
            "Subject": values[1],
            "Resolution": values[2]
        }
        parsed_data[cluster_name] = item_dict

    #parsed data to JSON

    parsed_json = json.dumps(parsed_data, indent=4)
   
    return parsed_data


data = pd.read_csv('Ticket_Data.csv')


#gdpr sanitization for Subject
data['Subject'] = data['Subject'].astype(str)
column_name = 'Subject'
data = remove_confidential_info(data, column_name)

#getting the embeddings of Subjects
sentences = data['Subject'].tolist()
sent_embedder = model_config()
embeddings = sent_embedder.encode(sentences)

