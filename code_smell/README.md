## Code Smell

### 0. Data

The code summarization dataset is at `data/code_summarization_data.jsonl`. We explain the fields of the data below:


|    field    |                 description                 |
| :----------: | :------------------------------------------: |
|      id      |     the local id of items in the dataset     |
| lang_cluster | the programming language of the source code |
| source_code |     code containing certain code smell     |
|  smell_code  |   smelly code snippet in the source code   |
|    smell    | code smell type of the smelly code snippet |

### 1. Installation

1. `cd code_smell`
2. install `python>=3.9` (we only guarantee the code works on python 3.9)
3. install `torch` (we suggest `torch==2.1.1`) based on your cuda version
4. `pip install -r requirements.txt`

### 2. Inference

Run the inference scripts to get the inference results of the targeted LLMs. The inference results `code_smell_result_{model_name}.jsonl` will be saved under the `inference/results` folder. The inference logs `code_smell_log_{model_name}.log` will be saved under the `inference/logs` folder.

#### 2.1 Closed-sourced LLMs

We provide the following closed-sourced LLMs inference scripts for you:


| Model Name | Model Version      | Script Name  |
| ---------- | ------------------ | ------------ |
| PaLM 2     | text-bison-001     | run_palm2.py |
| GPT-4      | gpt-4-0613         | run_gpt.py   |
| GPT-3.5    | gpt-3.5-turbo-0613 | run_gpt.py   |

For PaLM 2, you can run the following command by replacing `google_api_key` with your own Google API key.

`python inference/run_palm2.py --api_key google_api_key`

For GPT, you can run the following command by replacing `openai_api_key` with your own OpenAI API key, `model_version` with specific model version.

`python inference/run_gpt.py --api_key openai_api_key --model model_version`

#### 2.2 Open-sourced LLMs

We provide the following open-sourced LLMs inference scripts for you:


| Model Name | Model Checkpoint | Script Name |
| ---------- | ---------------- | ----------- |
|            |                  |             |
|            |                  |             |

For HuggingFace models, you can run the following command by replacing `huggingface_access_token` with your own HuggingFace access token, `cache_dir` with path to a directory in which a downloaded pretrained model and tokenizer should be cached, `model_checkpoint` with specific model checkpoint.

`python inference/run_{model_name}.py --access_token huggingface_access_token --cache_dir cache_dir --checkpoint model_checkpoint`

### 3. Evaluation

`cd inference`

#### 2.1 PaLM

Replace "google_api_key" with your own Google API key.

`python run_palm.py --api_key google_api_key --data_load_name code_summarization_data.jsonl --result_save_name code_summ_infer_palm.jsonl --log_file_name code_summ_infer_palm.log`

#### 2.2 GPT-3.5

Replace "openai_api_key" with your own OpenAI API key.

`python run_gpt.py --api_key openai_api_key --model gpt-3.5-turbo-0613 --data_load_name code_summarization_data.jsonl --result_save_name code_summ_infer_gpt3.jsonl --log_file_name code_summ_infer_gpt3.log`

#### 2.3 GPT-4

Replace "openai_api_key" with your own OpenAI API key.

`python run_gpt.py --api_key openai_api_key --model gpt-4-0613 --data_load_name code_summarization_data.jsonl --result_save_name code_summ_infer_gpt4.jsonl --log_file_name code_summ_infer_gpt4.log`

#### 2.4 Huggingface Models

For huggingface models, you can run the following command by replacing "access_token" with your own HuggingFace access token, "cache_dir" with path to a directory in which a downloaded pretrained model should be cached, "model_checkpoint" with the name of the targeted model, and "{model_name}" with the name of the targeted huggingface model's name.

`python run_{model_name}.py --access_token access_token --cache_dir cache_dir --checkpoint model_checkpoint --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_{model_name}.jsonl --log_file_name code_opt_infer_{model_name}.log`

We provide the following huggingface models inference scripts for you:


| Model Name |          Model Checkpoint          |    Script Name    |
| :---------: | :---------------------------------: | :----------------: |
|  codellama  | codellama/CodeLlama-34b-Instruct-hf |  run_codellama.py  |
|   vicuna   |      lmsys/vicuna-13b-v1.5-16k      |   run_vicuna.py   |
|   llama2   |   meta-llama/Llama-2-70b-chat-hf   |   run_llama2.py   |
| wizardcoder |    WizardLM/WizardCoder-15B-V1.0    | run_wizardcoder.py |
|  starcoder  |     HuggingFaceH4/starchat-beta     |  run_starcoder.py  |

An example of running the codellama model is:

`python run_codellama.py --access_token access_token --cache_dir cache_dir --checkpoint codellama/CodeLlama-34b-Instruct-hf --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_codellama.jsonl --log_file_name code_opt_infer_codellama.log`

### 3. Evaluation

1. `cd ../evaluator`
2. Run `python score_code_summarization.py --llm_infer_result infer_file` (replace "infer_file" with the llm's inference file name, eg. "code_summ_infer_palm.jsonl") to get the scores of the targeted llm's inference results, the scores will be palced at `evaluator/summ_scores/`.

Code Smell Task

## Feature Definitions

- `id`:
- `lang_cluster`:
- `source_code`:
- `smell_code`:
- `smell`:

### 0. Data

The code summarization dataset is at data/code_summarization_data.jsonl. We explain the fields of the data below:


| Field        | Description                                   |
| ------------ | --------------------------------------------- |
| id           | the local id of items in the dataset          |
| source_code  | a code snippet that perform some functionaliy |
| lang_cluster | the programming language of the source code   |


| field               | description                                               |
| ------------------- | --------------------------------------------------------- |
| id                  | the local id of items in the dataset                      |
| source_code         | a code snippet that perform some functionaliy             |
| lang_cluster        | the programming language of the source code               |
| human_summarization | a piece of reference nl summarization for the source code |
