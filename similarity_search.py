import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import time
import re

def get_embedding(text):
    sent_embedder = model_config()
    embeddings = sent_embedder.encode(text)
    return embeddings

def model_config():
    sent_emb_model = 'multi-qa-mpnet-base-dot-v1'
    sent_emb_max_seq_length = 500
    sent_embedder = SentenceTransformer(sent_emb_model, device = 'cpu')
    sent_embedder.max_seq_length = sent_emb_max_seq_length
    return sent_embedder


def remove_confidential_info(dataframe, column_name):
    # Define regular expressions to match email addresses and phone numbers
    
    email_pattern = r'\S+@\S+'
    phone_pattern = r'\+\d{10,12}'
    
    # Iterate through the specified column
    for index, row in dataframe.iterrows():
        text = row[column_name]
        
        # substitute email addresses
        text = re.sub(email_pattern, '{MASKED_ID}', text)
        
        # substitute phone numbers
        text = re.sub(phone_pattern, '{MASKED_NO.}', text)
        
        # Update the DataFrame with the cleaned text
        dataframe.at[index, column_name] = text
    
    return dataframe


historical_tickets = pd.read_csv('Ticket_Data.csv')
#gdpr sanitization for Resolution
historical_tickets['Resolution Description'] = historical_tickets['Resolution Description'].astype(str)
column_name = 'Resolution Description'
historical_tickets = remove_confidential_info(historical_tickets, column_name)

#gdpr sanitization for Subject
historical_tickets['Subject'] = historical_tickets['Subject'].astype(str)
column_name = 'Subject'
historical_tickets = remove_confidential_info(historical_tickets, column_name)

# Creating embeddings for historical ticket Suject
sentences = historical_tickets['Subject'].tolist()
sent_embedder = model_config()
embeddings = sent_embedder.encode(sentences)


# FAISS index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d) 
index.add(embeddings)

# Saving the FAISS index
#faiss.write_index(index, 'history_tickets_4.0.index')

#historical_tickets.to_csv('Ticket_Data.csv',index=False)

def calculate_similarity_scores(new_ticket_embedding, historical_embeddings):
    similarity_scores = cosine_similarity([new_ticket_embedding], historical_embeddings)
    return similarity_scores[0]


# Function to find top k similar tickets
def find_similar_tickets(new_ticket_text, k=5):
    new_ticket_embedding = get_embedding(new_ticket_text)
    D, I = index.search(np.array([new_ticket_embedding]), k)
    return I[0], embeddings


def get_similar_ticket_details(new_ticket_text, similar_ticket_indices, embeddings):
    
    # Getting the similar ticket subject details
    #historical_tickets = pd.read_csv('Ticket_Data.csv')
    historical_tickets = pd.read_csv('Ticket_Data.csv')
    #gdpr sanitization for Resolution
    historical_tickets['Resolution Description'] = historical_tickets['Resolution Description'].astype(str)
    column_name = 'Resolution Description'
    historical_tickets = remove_confidential_info(historical_tickets, column_name)
    #gdpr sanitization for Subject
    historical_tickets['Subject'] = historical_tickets['Subject'].astype(str)
    column_name = 'Subject'
    historical_tickets = remove_confidential_info(historical_tickets, column_name)
    
    
    similar_tickets = historical_tickets.iloc[similar_ticket_indices]
    new_ticket_embedding = get_embedding(new_ticket_text)
    similarity_scores = calculate_similarity_scores(new_ticket_embedding, embeddings)
    
    # Ranking Similarity Scores and sorting in descending order
    historical_tickets['Similarity_Score'] = similarity_scores
    historical_tickets = historical_tickets.sort_values(by='Similarity_Score', ascending=False)

    # Calculation of similarity rank
    historical_tickets['Similarity_Rank'] = range(1, len(historical_tickets) + 1)
    
    
    k=5
    results_dev = pd.DataFrame({
        'Similar_Ticket_ID': similar_tickets['Case Number'],
        'Similarity_Score': historical_tickets['Similarity_Score'][0:k],
        'Rank': historical_tickets['Similarity_Rank'][0:k],
        'Subject': similar_tickets['Subject'],
        'Resolution': similar_tickets['Resolution Description']
    })
    
    results_user = pd.DataFrame({
        'Rank': historical_tickets['Similarity_Rank'][0:k],
        'Subject': similar_tickets['Subject'],
        'Resolution': similar_tickets['Resolution Description']
    })
    
    results_dev = results_dev.sort_values(by='Rank',ascending=True)
    results_user = results_user.sort_values(by='Rank',ascending=True)

    return results_dev, results_user

