# Introduction
This code repository is for domain adaptation of single cell datasets.

# Code usage 
Prepare a data directory with the following structure:
```
data directory/
├── scDataset_1.dataset
├── scDataset_2.dataset
├── ...
└── scDataset_N.dataset
```
# Module
```
batch effect mitigation module/
├── pad_or_truncate_input_ids: Transform gene inputs to homogeneous size
├── Transformer Encoder (nhead=4, num_layers=2): Learning domain adapted embeddings $g(X)$.
├── Bacth Discriminator (input dim, output=n_batches): Train a discriminator $D$ to predict batch $B$ from $g(X)$. Simultaneously train $g$ to fool the discriminator (i.e., make $B$ unpredictable from $g(X)$).
├── CellTypeClassifier: Classification head to predict cell types.
├── Loss modules: Compute Maximum Mean Discrepancy (MMD) loss to explicitly match $P(g(X) \mid B_i)$ and $P(g(X) \mid B_j)$.
├── training_step: Train the model for one epoch.
├── training_model: Wrapper for training the model on dataset for epochs.
├── compute_embedding/extract_embedding: Using trained encoder $g(X)$ to output cell embedding. Compute Silhouette score, Calinski-Harabasz and Davies-Bouldin Index.
└── Visualization of cell embeddingn by batch and cell types. Output interactive plots.
```
# Example usage
```python
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
```
