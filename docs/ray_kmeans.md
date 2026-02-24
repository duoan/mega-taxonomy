# Ray K-means Clustering

## Overview

Ray K-means is a high-performance distributed K-means clustering system designed for mega-scale semantic segmentation. It features a revolutionary **zero-shuffle architecture** that eliminates traditional distributed computing bottlenecks, achieving 100x cost reduction and hours vs. weeks runtime compared to traditional Spark solution.

## Functionality

This clustering stage processes large-scale semantic embedding data stored in distributed file system (S3, GCS, Azure Blob, HDFS, etc) and performs distributed K-means clustering using Ray workers with accelerator devices (GPU, TPU, NPU, etc.). Key features include:

- **Data Streaming**: Direct DFS data streaming without inter-node data transfer.
- **Zero Shuffle**: Eliminates expensive data shuffling between iterations
- **Accelerator Support**: Custom accelerator kernel on CPU/GPU/TPU/ect with 5-8x speedup over native PyTorch.
- **Distributed Computing**: Ray-based parallel processing with auto-scaling.
- **Fault Tolerance**: Stateless workers enable easy recovery from failures.
- **Memory Efficient**: Streaming data processing for datasets larger than memory.
- **Auto-Configuration**: Accelerator device architecture dectection and auto-configuration.

## Architecture

The system uses a stateless worker design where:

- **Driver Node**: Coordinates training without touching data.
- **Ray Workers**: Process assigned input files independently.
- **Distributed Storage**: Central data repository for inputs, checkpoints and outputs.
- **No Data Shuffling**: Workers read data directly from distributed storage, eliminating network bottlenecks.

```text

Distributed Storage -> Ray Workers -> Accelerator Processing -> Results/Checkpoints -> Driver Aggregatationg
        ^                                                                                  |
        |                                                                                  V
        -----------------------------------------New Centroids------------------------------
```

## Parameters

### Core Parameters

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `d`        | int  | **Required** | Embedding dimension |
| `k`         | int  | **Required** | Number of clusters (centroids)|
| `input_path`| str | **Required** | Path to input embeddings (e.g., `s3://mybucket/prefix/`)|
| `output_path`| str | **Required** | Path to ouput results|
| `niter`|  int| `25` | Number of training iterations|
| `seed` | int | `42` | Random seed |
| `verbose` | bool | `true` | Enable detailed logging |

### Worker Configurations

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `n_workers`| int | Auto-detected | Number of Ray workers (auto-configured based on accelerators)|
| `num_gpus_per_worker` | float | `1` | GPU allocation per worker (e.g., `0.25` = 4 workers per GPU)|
| `num_cpus_per_worker` | float | `None` | CPU allocation per worker|
| `memory_mb_per_worker`|float | `None`| Memory allocation per worker (in MB)|

### DataLoader Configurations

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `dataloader_batch_size` | int | `10000` | Batch size for data loader|
| `dataloader_num_workers` | int | `4` | Number of Dataloader workers per Ray worker|
| `dataloader_prefetch_factor` | int | `2` | Number of batches to prefetch|

### Initialization Parameters

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `init_sample_size` | int | `100000` | Number of samples to use for initialization|

### Environment Configurations

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `envs` | dict | `None` | Environment configurations for each environment variable|

## YAML Configurations

### Basic Configurations

```yaml
name: "multimodal_clustering"
type: "clustering"
worker:
    number_gpus: 0.5
    number_workers: 2
    envs:
        CUDA_VISIBLE_DEVICES: "0,1"

input_path: "s3://s3_bucket_name/path/to/data"
output_path: "s3://s3_bucket_name/path/to/output"

engine:
    embedding_dim: 768
    n_clusters: 100000
    n_iterations: 25

```
