This repo explores the possibility of generating questions to fully evaluate Code-RAG systems.

Specifically, it focuses on the use of **knowledge graphs constructed from code repositories** to structure and query 
relevant information. From this foundation, we define **different categories of questions** that probe various 
dimensions of a RAG (Retrieval-Augmented Generation) system's performance and limitations.

These categories include, but are not limited to:

* **Chunk Count Reasoning**: Questions that test the system's ability to reason over a specific number of retrieved chunks.
* **Inter-Chunk Distance**: Questions designed to challenge how well the system connects related but distant pieces of information across code chunks.
* **Cross-File Dependencies**: Evaluation of whether the system can accurately follow logic or function definitions that span multiple files.
* **Semantic Linking**: Tests whether the system understands conceptual relationships (e.g., between classes, functions, and variables).

Each category includes **graded difficulty levels** to better assess the system’s capabilities. Difficulty is determined
based on factors like the number of hops in the knowledge graph, required cross-references, and ambiguity in the code context.

Ultimately, this approach aims to uncover **failure modes** and **comprehension gaps** in current Code-RAG systems, 
offering a structured path toward improvement through targeted evaluation.



# Using the code 
## The RepoKnowledgeGraph class 

`RepoKnowledgeGraph` is a Python class that builds a **knowledge graph representation of a source code repository**. It parses directories, files, and code chunks, extracts semantic relationships such as *calls* and *definitions*, and provides utilities for exploration, visualization, and embedding-based code indexing.

---

### 🚀 Features
* Parses a codebase into hierarchical nodes: directories, files, and code chunks.
* Constructs a **directed acyclic graph (DAG)** representing code structure.
* Parses directories, files, and code chunks into a semantic hierarchy.
* Uses LLM-based summarization and embedding for understanding code semantics.
* Detects **calls** and **defined entity** relationships.
* Allows for **graph visualization**, tree display, and JSON-based serialization.
* Compatible with downstream tasks such as **code search**, **analysis**, and **navigation**.

---

### 📦 Class Overview

```python
RepoKnowledgeGraph(path: str)
```

* Constructs the full knowledge graph from the root of a repository.
* Automatically extracts and embeds code chunks.
* Builds "contains" and "calls" relationships between nodes.

### Node Types

The definition of the classes can be found in `Node.py`.

* `Node`: Abstract base
* `DirectoryNode`: Represents folders
* `FileNode`: Represents individual source files
* `ChunkNode`: Represents code chunks (e.g., functions, classes)


### 📂 Basic Usage

#### 1. Construct from directory

```python
from RepoKnowledgeGraph import RepoKnowledgeGraph

graph = RepoKnowledgeGraph(path="/path/to/repo")
```

This will:

* Traverse the directory
* Parse files into chunks
* Embed chunks
* Detect entity relationships
* Build a knowledge graph

#### 2. Print the structure

```python
graph.print_tree()
```

#### 3. Save a visualization

```python
graph.save_graph_visualization("output.png")
```

#### 4. Access nodes and data

```python
for node in graph:
    print(node.name, node.node_type)
```

```python
file_node = graph['src/main.py']
print(file_node.defined_entities)
```

#### 5. Get neighbors or previous chunk

```python
neighbors = graph.get_neighbors("src/module.py_1")
previous = graph.get_previous_chunk(chunk_node)
```

---

### 💾 Save & Load Graph

#### Save to JSON file:

```python
graph.save_graph_to_file("graph.json")
```

#### Load from JSON file:

```python
from RepoKnowledgeGraph import RepoKnowledgeGraph

graph = RepoKnowledgeGraph.load_graph_from_file("graph.json")
```

---

### 🧠 Advanced Usage

#### Instantiate from dict (e.g., for API use):

```python
graph_dict = graph.to_dict()
new_graph = RepoKnowledgeGraph.from_dict(graph_dict)
```

#### Manual graph instantiation (without running `__init__`):

```python
graph = RepoKnowledgeGraph.from_path("/some/path")
```

---

### 📊 Visual Output Example

Edges:

* `contains` → green
* `calls` → blue

Each node displays:

* Name
* Called entities (top 3)
* Defined entities (summary)

Use `save_graph_visualization()` to output a `.png`.

---

### 🧱 Extensibility

You can plug in:

* Custom code chunking logic (via `CodeParser`)
* Custom LLM embeddings or summarization (via `ModelService`)
* Entity extractors suited for different programming languages

---

### 📎 Directory Tree Representation

Use:

```python
graph.print_tree(max_depth=3)
```

To limit the visual depth and focus on top-level structure.




## 🤖 Model Integration with `ModelService`

`RepoKnowledgeGraph` leverages LLMs for two core tasks:

### 1. **Code Summarization**

Used to generate meaningful `description` fields for each `ChunkNode`, enabling code search, documentation, and visualization.

### 2. **Code Embedding**

Used to create semantic vector embeddings for code chunks. These are stored in `ChunkNode.embedding` and support advanced features like:

* Code similarity search
* LLM retrieval-augmented generation (RAG)
* Chunk-level search and recommendations

---

### 🔧 Configuration via `ModelService`

The `ModelService` class is written to be as portable as possible. It can work with either local or deployed OpenAI servers. It suffices to set the correct environment variables.  

```python
from ModelService import ModelService

model = ModelService()
summary = model.query("Summarize this code:\n...")
embedding = model.embed("def foo(): ...")
```

---

### ⚙️ Environment Configuration

| Variable                | Description                               | Default                               |
| ----------------------- | ----------------------------------------- | ------------------------------------- |
| `OPENAI_BASE_URL`       | Base URL for the LLM (e.g., local server) | `http://0.0.0.0:8000/v1`              |
| `OPENAI_TOKEN`          | Auth token for LLM requests               | `no-need`                             |
| `MODEL_NAME`            | Model for summarization                   | `meta-llama/Llama-3.2-3B-Instruct`    |
| `OPENAI_EMBED_BASE_URL` | Base URL for embedding model              | `http://0.0.0.0:8001/v1`              |
| `OPENAI_EMBED_TOKEN`    | Token for embedding API                   | `no-need`                             |
| `EMBED_MODEL_NAME`      | Model for embeddings                      | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` |
| `STOP_AFTER_ATTEMPT`    | Retry count for LLM calls                 | `5`                                   |
| `WAIT_BETWEEN_RETRIES`  | Wait time between retries (in seconds)    | `2`                                   |

These settings can be provided via a `.env` file, which is automatically loaded with `dotenv`.

---

### 🛡️ Robustness

All LLM calls are **retried on failure** using `tenacity`:

* Retries up to `STOP_AFTER_ATTEMPT` times
* Waits `WAIT_BETWEEN_RETRIES` seconds between attempts
* Automatically logs failures and retries

---

### 🧪 Example `.env`

```env
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_TOKEN=your_token
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct

OPENAI_EMBED_BASE_URL=http://localhost:8001/v1
OPENAI_EMBED_TOKEN=your_token
EMBED_MODEL_NAME=Alibaba-NLP/gte-Qwen2-1.5B-instruct

STOP_AFTER_ATTEMPT=5
WAIT_BETWEEN_RETRIES=2
```

