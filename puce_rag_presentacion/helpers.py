"""
Created by Analitika at 12/08/2024
contact@analitika.fr
"""

# External imports
import os
import pandas as pd
import json
import requests
import torch
import plotly.graph_objects as go
import plotly.express as px

# Internal imports
from config import DATA_PATH, HUGGINGFACE_TOKEN


def load_sentences():
    src = os.path.join(DATA_PATH, "short_sentences.json")
    sens = pd.DataFrame()
    with open(src, "r", encoding="utf-8") as f:
        sens = pd.DataFrame(json.load(f))
    return sens


def compute_embeddings(text_, tokenizer, model):
    # Prepare text
    encoded_input = tokenizer(text_, return_tensors="pt")
    print(text_, len(encoded_input["input_ids"][0]))
    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Obtain the embeddings from the last hidden state
    return model_output.last_hidden_state.mean(dim=1).squeeze().numpy()


def query_embeddings(payload):
    model_embeddings = "sentence-transformers/all-MiniLM-L6-v2"
    url = f"https://api-inference.huggingface.co/models/{model_embeddings}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


def do_radar_plot(results, sentences_df, annotation):

    # Get the sentences (index) and their similarity values for the radar plot
    theta = [f"frase {_}" for _ in sentences_df.index]

    # Create radar plot
    fig = go.Figure()

    for col in sentences_df.columns:
        # Select the top sentences and corresponding similarities for each column
        top_sentences_i = results[col].tolist()
        # top_sentences_2 = results[sentences_df.columns[1]].tolist()
        # top_sentences_3 = results[sentences_df.columns[2]].tolist()

        # Add the first curve
        fig.add_trace(go.Scatterpolar(r=top_sentences_i, theta=theta, fill="toself", name=col))

    # Update the layout of the radar plot
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Medidas de similaridad semántica: <br>" + annotation + "<br>",
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="",
                showarrow=False,
                font=dict(size=12),
                align="left",
            )
        ],
    )

    # Show the plot
    fig.show()


def do_scatter_plot(df_emb):
    # Create a scatter plot with different symbols for each label
    fig = px.scatter(
        df_emb,
        x="X",
        y="Y",
        color="label",  # Different colors for different labels
        symbol="label",  # Different symbols for different labels
        title="UMAP Proyección de Embeddings",
        labels={"X": "UMAP Dimension 1", "Y": "UMAP Dimension 2"},
    )

    # Customize layout for better visualization
    fig.update_traces(marker=dict(size=10, line=dict(width=2)))
    fig.update_layout(legend_title_text="Categories de Frases", width=800, height=600)

    # Show the plot
    fig.show()
