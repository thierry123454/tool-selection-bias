<div align= "center">
    <h1> Tool-Selection Bias in LLMs </h1>
</div>

<p align="center">
  <a href="#quick-setup">Quick Setup</a> •
  <a href="#overall-structure">Overall Structure</a> •
  <a href="#citation">Citation</a>
</p>

</div>

This repo contains the code to reproduce our research on tool-selection bias: the tendency of LLMs to prefer some APIs over functionally equivalent alternatives. It includes:
- A bias benchmark (10 clusters × 5 APIs × 100 queries),
- Experiments showing API/position bias across 7 model families,
- Feature-level analysis & metadata perturbations,
- A biased continued pre-training (CPT) study,
- A lightweight subset-selection mitigation that reduces bias.

Built on top of [ToolBench / ToolLLM](https://github.com/OpenBMB/ToolBench). Please also see their license and citation.

Here is an overview of the different phases in how we measure and aim to understand tool-selection bias.
First, we embed and cluster the existing APIs in ToolLLM, then generate
queries for each cluster such that each API within the cluster can satisfy the query to
create our bias-evaluation benchmark. We run inference on this benchmark
using various models, compute the empirical selection distributions and our
bespoke bias metric, and finally investigate why models exhibit particular
biases via a range of different experiments.

<br>
<div align="center">
<img src="assets/overview_bias.png" width="800px">
</div>
<br>

<h2 id="quick-setup">🚀 Quick Setup</h2>

### Prerequisites
- Python: 3.10+
- GPU: CUDA-enabled GPU  with working drivers if you want to use local models.

### Clone the repo

```
git clone https://github.com/thierry123454/tool-selection-bias.git
cd tool-selection-bias
```

### System essentials
```
sudo apt update && sudo apt upgrade -y
sudo apt install -y git build-essential cmake curl
```
### Python environment
```
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Optional extras you may use:
pip install torch torchvision anthropic
```

If you hit binary issues with bitsandbytes/triton, you can remove them:
```
pip uninstall -y bitsandbytes triton
```

If you see HF hub import/version errors, pin:
```
pip install "huggingface-hub==0.11.1"

```

### Download ToolLLM data

Please download the ToolLLM dataset using the following link [Google Drive](https://drive.google.com/uc?id=1vzUpO2TadV97upKwLn-TWHA-PR57Vs2H).
*Please make sure you have downloaded the necessary data and put the directory (e.g. `data/`) under `ToolBench/`, so that the following bash scripts can navigate to the related data.*
Via gdown:
```
pip install gdown
gdown https://drive.google.com/uc?id=1vzUpO2TadV97upKwLn-TWHA-PR57Vs2H -O data.zip
unzip data.zip
```

### Bias queries (provided)
```
mkdir -p data_bias/instruction
mv 3_generate_queries_for_clusters/toolbench_bias_queries.json \
   data_bias/instruction/
```

### Environment variables
Set keys for whichever providers you’ll use (leave unset if you’re running locally without live APIs):

```
# RapidAPI (if calling the live ToolBench/rapid server yourself)
export RAPIDAPI_KEY="YOUR_RAPIDAPI_KEY"

# OpenAI (ChatGPT family)
export OPENAI_KEY="YOUR_OPENAI_KEY"

# Google (Gemini)
export GEMINI_KEY="YOUR_GEMINI_KEY"

# Same for any other LLM API

export PYTHONPATH="$(pwd)"
# (Optional) pick a single GPU
export CUDA_VISIBLE_DEVICES=0

```

⸻

### ▶️ Run

#### ToolLLaMA (local model, recommended baseline)
```
python toolbench/inference/qa_pipeline.py \
  --tool_root_dir data/toolenv/tools/ \
  --backbone_model toolllama \
  --model_path ToolBench/ToolLLaMA-2-7b-v2 \
  --max_observation_length 1024 \
  --method CoT@1 \
  --input_query_file data_bias/instruction/toolbench_bias_queries.json \
  --output_answer_file data_bias/answer_toolllama \
  --rapidapi_key $RAPIDAPI_KEY \
  --use_rapidapi_key \
  --test_bias
```

#### ChatGPT (OpenAI)
```
python toolbench/inference/qa_pipeline.py \
  --tool_root_dir data/toolenv/tools/ \
  --backbone_model chatgpt \
  --openai_key $OPENAI_KEY \
  --max_observation_length 1024 \
  --method CoT@1 \
  --input_query_file data_bias/instruction/toolbench_bias_queries.json \
  --output_answer_file data_bias/answer_chatgpt_no_func_base_prompt \
  --rapidapi_key $RAPIDAPI_KEY \
  --use_rapidapi_key \
  --test_bias
```

#### Gemini
```
python toolbench/inference/qa_pipeline.py \
  --tool_root_dir data/toolenv/tools/ \
  --backbone_model gemini \
  --openai_key $GEMINI_KEY \
  --max_observation_length 1024 \
  --method CoT@1 \
  --input_query_file data_bias/instruction/toolbench_bias_queries.json \
  --output_answer_file data_bias/answer_gemini \
  --rapidapi_key $RAPIDAPI_KEY \
  --use_rapidapi_key \
  --test_bias
```

#### General 
```
python toolbench/inference/qa_pipeline.py \
  --tool_root_dir data/toolenv/tools/ \
  --backbone_model {qwen-235b}/{deepseek}/{claude} \
  --openai_key ${LLM_KEY} \
  --max_observation_length 1024 \
  --method CoT@1 \
  --input_query_file data_bias/instruction/toolbench_bias_queries.json \
  --output_answer_file data_bias/{answer_dir} \
  --rapidapi_key $RAPIDAPI_KEY \
  --use_rapidapi_key \
  --test_bias
```

Note:
- --test_bias only extracts the first endpoint call and stops execution afterwards

<h2 id="overall-structure">📦 Overall Structure</h2>

### Dataset Generation

Below are the dataset stats used to test tool-selection bias:

| Clusters | API / Cluster | Queries / Cluster |
|-----------|----------|----------------|
| 10      | 5    | 100         |

The pipeline to generate it has three stages:
1. Collect & embed endpoint metadata
2. Form functionally-equivalent clusters & refine
3. Generate queries & export into ToolBench-format with controlled API ordering

Auth note: Some scripts use the legacy OpenAI SDK (OPENAI_API_KEY), others the new client (OPENAI_KEY). To be safe, set both:

```
export OPENAI_API_KEY="sk-..."   # legacy SDK
export OPENAI_KEY="sk-..."       # new SDK
```

⸻

#### Endpoint metadata & embeddings (folder: *1_endpoint_metadata_and_embed/*)
##### extract_api_metadata.py
Parses data/toolenv/tools/**/*.json and writes a compact map of tools → (tool_desc, [api_name, api_desc]).
- In: data/toolenv/tools/ (from ToolLLM)
- Out: api_metadata.json
<br>
Run:

```
cd 1_endpoint_metadata_and_embed
python extract_api_metadata.py
```

##### create_embeddings_openai.py
Builds texts like "Tool: <tool_desc> | <api_name>: <api_desc>" and embeds them with text-embedding-ada-002.
- In: api_metadata.json
- Out: embeddings_combined_openai.npy
<br>
Run:

```
python create_embeddings_openai.py

```

⸻

#### Cluster generation & refinement (folder: *2_generate_clusters_and_refine/*)
##### generate_duplicate_clusters.py
Given a seed list of “general” APIs, finds top-K nearest neighbors in embedding space, then iteratively removes outliers via an LLM check (ensures all endpoints can satisfy the same task). Also pulls required params from ToolLLM queries for context.
- In:
- - ../1_endpoint_metadata_and_embed/api_metadata.json
- - ../1_endpoint_metadata_and_embed/embeddings_combined_openai.npy
- - ../data/instruction/G1_query.json (to attach required params)
- Out: duplicate_api_clusters_2.json (rename to your preferred canonical filename)
<br>
Run:

```
cd ../2_generate_clusters_and_refine
python generate_duplicate_clusters.py
```

##### refine_retrieval_clusters.py (optional, interactive)
Suggests nearest neighbors for undersized clusters; you can add items by number.
- In: duplicate_api_clusters.json (set this path in the script)
- Out: updated duplicate_api_clusters.json
<br>
Run:

```
python refine_retrieval_clusters.py
```
⸻

#### Query generation & export (folder: *3_generate_queries_for_clusters/*)

You can generate free-text queries directly, or via template filling (good when open-ended generation drifts toward specific providers).

##### generate_queries_for_cluster.py
LLM produces 100 realistic queries per cluster that every endpoint can satisfy.
- In: ../2_generate_clusters_and_refine/duplicate_api_clusters.json
- Out: cluster_queries_3.json (one entry per cluster with 100 queries)
<br>
Run

```
cd ../3_generate_queries_for_clusters
python generate_queries_for_cluster.py
```

##### template_generation_for_cluster.py (optional)
Fills hand-written templates repeatedly to avoid provider-specific drift.
- In:
- - ../2_generate_clusters_and_refine/duplicate_api_clusters.json
- - templates.json (by cluster id)
- Out: filled_queries_by_template.json
<br>
Run

```
python template_generation_for_cluster.py
```

##### create_toolbench_format_dataset.py
Converts clusters + queries into ToolBench format, and controls API ordering per query. With SHUFFLE="cycle", each query is emitted once per API with a cyclic rotation. This is crucial to compensate for positional bias. You can limit clusters via HOLDOUT (e.g., [6,8,9,10]).
- In:
- - ../2_generate_clusters_and_refine/duplicate_api_clusters.json
- - cluster_queries.json
- - ToolLLM originals: ../data/instruction/G{1,2,3}_query.json (to fetch canonical API defs)
- - Fallbacks from data/toolenv/tools if needed
- Out: toolbench_bias_queries_cycle.json
- - With 10 clusters × 5 APIs × 100 queries and SHUFFLE="cycle", expect 5000 entries.
- - If HOLDOUT is set, only those clusters are emitted.
<br>
Run:

```
python create_toolbench_format_dataset.py
```

Tip: If you want exactly 100 prompts per cluster without rotations, set SHUFFLE="none" inside the script.

### Analyzing Data
The scripts to analyze the data output by the models are given in *4_gather_and_visualize_data*. This folder contains the post-processing pipeline for turning raw inference outputs into bias metrics and figures. Run *extract_selected_api.py* to extract the endpoints that were called and the positions of those endpoints in the API list.

### Bias Investigation
The folder *5_bias_investigation* holds the code that explains and probes tool-selection bias. It builds per-endpoint feature tables (e.g., query–metadata similarity, lengths, params, readability, age), runs the feature-level analysis (correlations and per-model linear regressions), and saves compact plots. It also contains code to generate metadata perturbations (scrambling/swapping of tool names). Lastly, it contains the biased continued pre-training (CPT) pipeline, where one can train an LLM with a corpus saturated in one endpoint’s metadata and evaluate how exposure changes selection shares.

### Bias Mitigation
The folder *6_bias_mitigation* contains the code to evaluate our subset-selection debiasing pipeline. It has code to synthesize a benchmark where each query is paired with 8 candidate APIs, of which a subset is truly sufficient; ground truth is saved alongside the dataset. We can run the selector to predict subsets, then use the evaluator to compute micro-precision, micro-recall, and exact-set match overall. The resulting filter is intended to precede uniform sampling among retained tools, flattening selection distributions without harming coverage.

<h2 id="citation">📚 Citation</h2>

Feel free to cite ToolBench. The paper for this research is still under development.
```bibtex
@misc{qin2023toolllm,
      title={ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs}, 
      author={Yujia Qin and Shihao Liang and Yining Ye and Kunlun Zhu and Lan Yan and Yaxi Lu and Yankai Lin and Xin Cong and Xiangru Tang and Bill Qian and Sihan Zhao and Runchu Tian and Ruobing Xie and Jie Zhou and Mark Gerstein and Dahai Li and Zhiyuan Liu and Maosong Sun},
      year={2023},
      eprint={2307.16789},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
