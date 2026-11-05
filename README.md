# Adaptive LLM Routing for Multi-Turn Conversations with Continuously Evolving User Queries

## Repository Structure

- `src/env.py`, `src/dialogue.py`, `src/task.py`, `src/utils.py`: Dialogue environment, state representation, task definitions, and data I/O.
- `src/run_mcts.py`, `src/mcts.py`: MCTS search and trajectory generation.
- `src/build_search_data.py`, `src/build_emb.py`: Search result processing and retrieval embedding construction.
- `src/train.py`, `src/model.py`: Router training, testing, and model definition.
- `data/{task}_{train,test}.jsonl`: Raw training and test data for each task.

## Environment Setup

We recommend using Conda to manage your Python environment:

```bash
conda create -n convorouter python=3.10
conda activate convorouter
python -m pip install -r requirements.txt
```

## API and LLM Pool Configuration

1. **API configurations and LLM pools.** Define `API` (service endpoints and keys) and `model_list` (candidate LLMs and their providers) in both `src/run_mcts.py` and `src/train.py`. Keep the two configurations consistent.

```python
API = {
    "API_NAME": {
        "base_url": "",
        "api_key": ""
    },
    # Used for calling the user simulator, evaluator, and candidate LLMs
}

model_list = {
    # Define your own LLM pools here.
    # Each pool can include multiple API providers and corresponding models.
    'LLM_Pool_Name': {
        'type': ['API_NAME1', 'API_NAME2', 'API_NAME3'],
        'model': ['LLM1', 'LLM2', 'LLM3'],
    },
}
```

2. **Response interface.** Adapt the chat request and response formats in `src/env.py`'s `query()` to the chosen LLM provider.

3. **Cost reward (optional).** Set the input prices `input_cost`, cached input prices `input_cache`, and output prices `output_cost` in `src/mcts.py` in candidate-LLM order. Enable the cost reward with `--lambda1` in `src/run_mcts.py`.

## Running the Pipeline

Run search, data construction, and router training and testing in sequence. In the commands below, `TASK_TYPE` is `sharegpt`, `jddc`, or `meddg`.

### Generate Search Trajectories

Run MCTS on the training set of the selected task to generate search trajectories.

```bash
python src/run_mcts.py --task TASK_TYPE --system_set MY_POOL --user_type USER_API --eval_type EVAL_API --user_model USER_MODEL --eval_model EVAL_MODEL
```

The script creates `results/{run_id}/`. The `train/` directory stores dialogue trajectories, `mcts/` stores search statistics, and `realization/` stores cached dialogue states.

### Construct Training Data and Retrieval Embeddings

Process the search results into supervised examples, then compute embeddings for state retrieval.

```bash
python src/build_search_data.py --task TASK_TYPE --system_set MY_POOL --run_id RUN_ID
python src/build_emb.py --task TASK_TYPE --system_set MY_POOL
```

Replace `RUN_ID` with the name of the search output directory. The processed training data is saved to `data/MY_POOL/TASK_TYPE_train.json`, and the retrieval embeddings to `data/MY_POOL/emb/TASK_TYPE_train_emb.pkl`.

### Train and Test the Router

Train the router on the constructed data and evaluate its dialogue performance during training.

```bash
python src/train.py --task TASK_TYPE --system_set MY_POOL --user_type USER_API --eval_type EVAL_API --user_model USER_MODEL --eval_model EVAL_MODEL
```

After each epoch, the script evaluates on the corresponding test set and saves test dialogues, model weights, and cached dialogue states to the new `results/{run_id}/test/`, `checkpoints/`, and `realization/` directories, respectively.
