import os
import subprocess

import torch

'''
if torch.cuda.is_available():
    print("CUDA is available.")
    allowed_devices = [1, 2, 3, 7]
    
    for dev in allowed_devices:
        if dev < torch.cuda.device_count():
            try:
                torch.cuda.set_device(dev)
                device = torch.device(f"cuda:{dev}")
                print(f"Using device: {device} - {torch.cuda.get_device_name(device)}")
                break
            except Exception as e:
                print(f"Failed to set device cuda:{dev} - {e}")
    else:
        device = torch.device("cpu")
        print("None of the preferred CUDA devices are available, falling back to CPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")
'''

os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import glob
import datetime
import numpy as np
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from collections import defaultdict
from geneformer import Classifier, EmbExtractor
from datasets import concatenate_datasets
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import tqdm
import pickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap


random.seed(42)
GPU_NUMBER = [1,2,3,7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



project_names = ['scRNA_v2_PMID32971526', 
                 'scRNA_v3_PMID32971526',
                 'snRNA_v2_PMID32971526',
                 'snRNA_v3_PMID32971526']
data_dirs = [f'data/{project}.dataset' for project in project_names]

# Set random seed for reproducibility (optional)



pid = 3
data_dir = data_dirs[pid]
dataset = load_from_disk(data_dir)
n_batches = np.unique(dataset['batch'], return_counts = True)
cell_types = np.unique(dataset['celltype'])

output_dir = f'eval-512-be_handing-GF95M-{project_names[pid]}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(dataset)
def pad_or_truncate_input_ids(input_ids, max_len=2048, pad_val=0):
        """
        Pads or truncates each input_ids list to fixed length max_len.
        """
        processed = []
        for seq in input_ids:
            if len(seq) < max_len:
                # pad with pad_val
                padded = seq + [pad_val] * (max_len - len(seq))
            else:
                # truncate to max_len
                padded = seq[:max_len]
            processed.append(padded)
        return np.array(processed)
# ----------------- MODEL COMPONENTS --------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead=4, num_layers=2, max_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)  # (B, L, D)
        x = self.transformer(x)  # (B, L, D)
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)
        return x


class BatchDiscriminator(nn.Module):
    def __init__(self, input_dim, n_batches):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_batches)
        )

    def forward(self, x):
        return self.net(x)

class CellTypeClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

# ----------------- MMD LOSS --------------------
def compute_mmd(x1, x2):
    xx, yy, zz = torch.mm(x1, x1.t()), torch.mm(x2, x2.t()), torch.mm(x1, x2.t())
    rx = xx.diag().unsqueeze(1)
    ry = yy.diag().unsqueeze(1)
    K = torch.exp(-0.5 * (rx + rx.t() - 2 * xx))
    L = torch.exp(-0.5 * (ry + ry.t() - 2 * yy))
    P = torch.exp(-0.5 * (rx + ry.t() - 2 * zz))
    beta = 1. / (x1.size(0) * x1.size(0))
    gamma = 1. / (x2.size(0) * x2.size(0))
    delta = 2. / (x1.size(0) * x2.size(0))
    return beta * K.sum() + gamma * L.sum() - delta * P.sum()

# ----------------- TRAINING STEP --------------------
def training_step(batch, encoder, classifier, discriminator, optimizer_e, optimizer_d, lambda_adv=1.0, lambda_mmd=1.0):
    input_ids, cell_labels, batch_labels = batch  # assumed tensorized & on device
    embeddings = encoder(input_ids)

    # Task Loss
    pred_cell = classifier(embeddings)
    task_loss = F.cross_entropy(pred_cell, cell_labels)

    # Adversarial Batch Loss
    pred_batch = discriminator(embeddings.detach())
    adv_loss_d = F.cross_entropy(pred_batch, batch_labels)

    # Train discriminator
    optimizer_d.zero_grad()
    adv_loss_d.backward()
    optimizer_d.step()

    # Fool discriminator
    pred_batch = discriminator(embeddings)
    adv_loss_g = -F.cross_entropy(pred_batch, batch_labels)

    # MMD Loss (optional)
    mmd_loss = torch.tensor(0.0, device=embeddings.device)
    for i in range(batch_labels.max().item() + 1):
        for j in range(i+1, batch_labels.max().item() + 1):
            xi = embeddings[batch_labels == i]
            xj = embeddings[batch_labels == j]
            if len(xi) > 1 and len(xj) > 1:
                mmd_loss += compute_mmd(xi, xj)

    # Total Loss
    total_loss = task_loss + lambda_adv * adv_loss_g + lambda_mmd * mmd_loss

    optimizer_e.zero_grad()
    total_loss.backward()
    optimizer_e.step()

    return total_loss.item(), task_loss.item(), adv_loss_d.item(), mmd_loss.item()

# ----------------- TRAINING LOOP --------------------
def train_model(dataset, encoder, classifier, discriminator, n_epochs=10, batch_size=2048, lr=1e-4):
    le_batch = LabelEncoder()
    le_cell = LabelEncoder()
    input_ids = dataset['input_ids']
    X_input = pad_or_truncate_input_ids(input_ids, max_len=2048)
    input_ids = torch.tensor(X_input, dtype=torch.long)
    batch_labels = torch.tensor(le_batch.fit_transform(dataset['batch']), dtype=torch.long)
    cell_labels = torch.tensor(le_cell.fit_transform(dataset['celltype']), dtype=torch.long)

    dataset_tensor = TensorDataset(input_ids, cell_labels, batch_labels)
    loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    encoder.to(device)
    classifier.to(device)
    discriminator.to(device)

    optimizer_e = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in tqdm.tqdm(range(n_epochs)):
        total_loss, task_loss, adv_loss, mmd = 0, 0, 0, 0
        for batch in tqdm.tqdm(loader):
            batch = [x.to(device) for x in batch]
            l, t, a, m = training_step(batch, encoder, classifier, discriminator, optimizer_e, optimizer_d)
            total_loss += l
            task_loss += t
            adv_loss += a
            mmd += m
        print(f"Epoch {epoch+1}: Total={total_loss:.4f}, Task={task_loss:.4f}, Adv={adv_loss:.4f}, MMD={mmd:.4f}")
        torch.save(encoder.module.state_dict(), os.path.join(output_dir, f"encoder_{project_names[pid]}_epoch-{epoch}.pt"))
        torch.save(discriminator.module.state_dict(), os.path.join(output_dir, f"discriminator_{project_names[pid]}_epoch-{epoch}.pt"))

def compute_embeddings(dataset, encoder, output_dir, project_name, batch_size=128):
    print(f"Computing embeddings for {project_name}...")
    input_ids = dataset['input_ids']
    X_input = pad_or_truncate_input_ids(input_ids, max_len=2048)
    input_ids_tensor = torch.tensor(X_input, dtype=torch.long)
    
    encoder.eval()
    encoder.to(device)
    
    all_embeddings = []
    loader = DataLoader(input_ids_tensor, batch_size=batch_size)

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            batch = batch.to(device)
            emb = encoder(batch)
            all_embeddings.append(emb.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    save_path = os.path.join(output_dir, f"embeddings_{project_name}.pt")
    torch.save(embeddings_tensor, save_path)
    print(f"Saved embeddings to {save_path}")

def extract_embeddings(encoder, dataset, max_len=2048, batch_size=256):
    encoder.eval()
    encoder.to(device)

    input_ids = pad_or_truncate_input_ids(dataset['input_ids'], max_len=max_len)
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    dataloader = DataLoader(input_ids, batch_size=batch_size)
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting embeddings"):
            batch = batch.to(device)
            emb = encoder(batch).cpu().numpy()
            all_embeddings.append(emb)

    return np.vstack(all_embeddings)

def process_input_ids_embedding_from_dataset(dataset, encoder, output_dir, n_pca=50, n_clusters=10, tsne_perplexity=30):
    # Step 1: Get embeddings from encoder
    print("Generating embeddings from model...")
    embeddings = extract_embeddings(encoder, dataset)

    # Step 2: PCA
    print("Computing PCA...")
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(embeddings)

    # Step 3: UMAP
    print("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    X_umap = reducer.fit_transform(X_pca)

    # Step 4: t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    # Step 5: Evaluation
    cluster_labels = dataset['celltype']

    def evaluate(X, name):
        print(f"\n{name} Embedding Quality:")
        sil = silhouette_score(X, cluster_labels)
        ch = calinski_harabasz_score(X, cluster_labels)
        db = davies_bouldin_score(X, cluster_labels)
        print(f"  Silhouette Score: {sil:.4f}")
        print(f"  Calinski-Harabasz Index: {ch:.2f}")
        print(f"  Davies-Bouldin Index: {db:.4f}")
        return {
            "embedding": name,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db
        }
    '''
    results = [
        evaluate(X_pca, "PCA"),
        evaluate(X_umap, "UMAP"),
        evaluate(X_tsne, "t-SNE")
    ]
    '''
    # Step 6: Save
    print("Saving embeddings and labels...")
    df = pd.DataFrame({
        'batch': dataset['batch'],
        'celltype': dataset['celltype']
    })

    df_umap = pd.DataFrame(X_umap, columns=['umap_1', 'umap_2'])
    df_pca = pd.DataFrame(X_pca, columns=[f'pca_{i+1}' for i in range(X_pca.shape[1])])
    df_tsne = pd.DataFrame(X_tsne, columns=['tsne_1', 'tsne_2'])

    full_df = pd.concat([df, df_pca, df_umap, df_tsne], axis=1)
    csv_path = os.path.join(output_dir, f"embeddings_with_metrics_{project_names[pid]}.csv")
    full_df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    return full_df


# ----------------- MODEL INSTANTIATION & TRAINING --------------------


token_dictionary_file = 'Geneformer/geneformer/token_dictionary_gc95M.pkl'
with open(token_dictionary_file, 'rb') as f:
    token_dict = pickle.load(f)
vocab_size = len(token_dict)
print("Vocab size:", vocab_size)
d_model = 512
encoder = TransformerEncoder(vocab_size, d_model=d_model)
classifier = CellTypeClassifier(input_dim=d_model, n_classes=len(cell_types))
discriminator = BatchDiscriminator(input_dim=d_model, n_batches=len(n_batches[0]))
encoder = nn.DataParallel(encoder)
classifier = nn.DataParallel(classifier)
discriminator = nn.DataParallel(discriminator)

train_model(dataset, encoder, classifier, discriminator)
emds = compute_embeddings(dataset, encoder, output_dir, project_names[pid])
embedding_df = process_input_ids_embedding_from_dataset(dataset, encoder, output_dir)
print(embedding_df)