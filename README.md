# Mega-Taxonomy

**Mega-Taxonomy** is a high-performance, web-scale distributed engine designed to organize the world’s largest multimodal datasets. It serves as the hierarchical "classification brain" for the [**Mega-Data-Factory**](https://github.com/duoan/mega-data-factory) ecosystem, capable of partitioning **200B+ samples** (1024D) into a structured, navigable semantic hierarchy.

## 🌟 Overview

At a scale of 200 billion vectors, traditional flat clustering methods collapse under communication overhead and computational complexity. **Mega-Taxonomy** solves this by implementing a distributed hierarchical KMeans strategy. By leveraging **Ray** for orchestration and custom **Triton kernels** for GPU acceleration, it transforms raw embeddings into a deterministic taxonomic tree, enabling search efficiency and massive-scale data governance.

---

## 🏗 System Architecture

Mega-Taxonomy utilizes a decoupled **Driver-Worker-DFS** architecture to handle trillion-scale operations:

* **Orchestration (Ray):** The central Driver manages cluster resources, tracks global convergence, and dispatches data partition URLs to distributed workers.
* **Hardware Acceleration (Triton):** Workers execute high-performance **Triton Kernels** to compute distance matrices. These kernels are manually optimized for 1024D float32/bfloat16 arithmetic, saturating Tensor Core throughput on NVIDIA GPUs.
* **State Persistence (DFS):** All centroids and tree nodes are persisted in a **Distributed File System** (S3/HDFS/Lustre). This allows the system to handle  values in the millions without being limited by the memory of a single node.

---

## ⚙️ Technical Workflow

1. **Initialization:** The Driver initializes root centroids and writes them to the DFS.
2. **Dynamic Dispatch:** Ray workers pull data partition URLs (sharded by Mega-Data-Factory).
3. **The "Triton-Iterate" Loop:**
* **Pull:** Workers fetch the latest centroids from DFS.
* **Compute:** Custom Triton kernels assign 200B samples to the nearest centroids.
* **Partial Reduce:** Workers compute local partial sums and counts.


4. **Global Synchronization:** The Driver aggregates partials from all workers to update the global centroids.
5. **Tree Evolution:** Once a level converges, the system recursively triggers the next level of partitioning, building the **Hierarchical Tree**.

---

## ⚡ Key Specifications

| Feature | Specification |
| --- | --- |
| **Data Scale** | 200 Billion+ Vectors |
| **Vector Dimension** | 1024D (Multimodal) |
| **Backend** | Ray (Distributed) + Triton (GPU Kernel) |
| **Complexity** |  via Hierarchical Indexing |
| **Storage** | DFS-backed (S3 / HDFS / Lustre) |

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Ray Cluster (2.7+)
* NVIDIA GPUs (Ampere architecture or newer recommended)
* Distributed File System access

### Installation

```bash
pip install mega-taxonomy
```

### Basic Usage

```python
from mega_taxonomy import TaxonomyEngine

# Configuration for 200B scale
engine = TaxonomyEngine(
    n_levels=5,
    branching_factor=1_000_000,
    dim=1024,
    storage_uri="s3://mega-factory/taxonomy-indices/"
)

# Launch distributed fit via Ray
engine.fit(
    input="s3://mega-factory/200B_embeddings_input/*.parquet",
    output="s3://mega-factory/200B_embeddings_output/",
)

# Generate hierarchical paths for your samples
paths = engine.predict("s3://mega-factory/new_samples/*.parquet")

```

---

## 🛠 Features & Roadmap

* [ ] **Distributed Ray Driver-Worker implementation.**
* [ ] **Custom Triton Kernels** for optimized 1024D Euclidean distance.
* [ ] **DFS Centroid State Management** for extreme .
* [ ] **HNSW-based Centroid Search** for even faster level-transitions.
* [ ] **Auto-Balancing:** Dynamic node splitting for skewed data distributions.

---

## 🤝 Part of the Mega-Suite

Mega-Taxonomy is designed to consume output directly from [**Mega-Data-Factory**](https://github.com/duoan/mega-data-factory). Together, they form a complete pipeline for processing, indexing, and understanding multimodal data at a planetary scale.

---
## Project Docs

For how to install uv and Python, see [installation.md](docs/installation.md).

For development workflows, see [development.md](docs/development.md).

For instructions on publishing to PyPI, see [publishing.md](docs/publishing.md).

* * *

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
