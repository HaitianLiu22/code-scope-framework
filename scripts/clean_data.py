import uuid
import warnings
import numpy as np

from pathlib import Path
from datasets import load_dataset


def update_testcases(example):
    testcases = eval(str(example['testcases']))
    for testcase in testcases:
        if 'time' in testcase.keys():
            del testcase['time']
        if 'mem' in testcase.keys():
            del testcase['mem']
        if 'exec_outcome' in testcase.keys():
            del testcase['exec_outcome']
        if 'result' in testcase.keys():
            del testcase['result']
    example['testcases'] = str(testcases)

    return example


def update_metrics(example):
    example['human_sample_pass_rate'] = np.mean([example[f'human_sample_pass_rate_{index + 1}'] for index in range(5)])
    example['human_sample_line_coverage'] = np.mean([example[f'human_sample_line_coverage_{index + 1}'] for index in range(5)])
    example['human_sample_branch_coverage'] = np.mean([example[f'human_sample_branch_coverage_{index + 1}'] for index in range(5)])

    return example


def main():
    load_data_name_list = [
        'code_summarization_data.jsonl',
        'code_smell_data.jsonl',
        'code_review_data.jsonl',
        'automated_testing_data.jsonl',
        'program_synthesis_data.jsonl',
        'code_translation_data.jsonl',
        'code_repair_data.jsonl',
        'code_optimization_data.jsonl'
    ]

    lang_cluster_mapping = {
        'c': 'C',
        'c#': 'C#',
        'c++': 'C++',
        'd': 'D',
        'delphi': 'Delphi',
        'go': 'Go',
        'java': 'Java',
        'javascript': 'JavaScript',
        'kotlin': 'Kotlin',
        'perl': 'Perl',
        'php': 'PHP',
        'python': 'Python',
        'ruby': 'Ruby',
        'rust': 'Rust'
    }

    for load_data_name in load_data_name_list:
        load_data_path = Path(__file__).parent.parent / Path('raw') / Path(load_data_name)
        save_data_path = Path(__file__).parent.parent / Path('data') / Path(load_data_name)
        dataset = load_dataset('json', split='train', data_files=str(load_data_path))
        dataset.cleanup_cache_files()
        print(dataset)

        if load_data_name == 'code_summarization_data.jsonl':
            code_uid_list = [str(uuid.uuid4()).replace('-', '') for _ in range(len(dataset))]
            dataset = dataset.add_column('code_uid', code_uid_list)
            dataset = dataset.remove_columns('task_name')
            dataset = dataset.remove_columns('task_url')
            dataset = dataset.remove_columns('task_cat')
            dataset = dataset.remove_columns('explain')
            dataset = dataset.remove_columns('LOC')
            dataset = dataset.rename_column('lang', 'lang_cluster')
            dataset = dataset.rename_column('code', 'source_code')
            dataset = dataset.rename_column('code_sum_groundtruth', 'human_summarization')
        elif load_data_name == 'code_smell_data.jsonl':
            dataset = dataset.remove_columns('difficulty')
            dataset = dataset.remove_columns('start_line')
            dataset = dataset.remove_columns('end_line')
            dataset = dataset.remove_columns('length')
        elif load_data_name == 'code_review_data.jsonl':
            dataset = dataset.remove_columns('length')
            dataset = dataset.rename_column('old_code', 'source_code')
        elif load_data_name == 'automated_testing_data.jsonl':
            dataset = dataset.remove_columns('prob_desc_memory_limit')
            dataset = dataset.remove_columns('difficulty')
            dataset = dataset.remove_columns('prob_desc_time_limit')
            dataset = dataset.remove_columns('prob_desc_input_from')
            dataset = dataset.remove_columns('prob_desc_output_to')
            dataset = dataset.remove_columns('length')
            dataset = dataset.remove_columns('num_hidden_unit_tests')
            dataset = dataset.remove_columns('lang')
            dataset = dataset.rename_column('prob_desc_sample_inputs', 'sample_inputs')
            dataset = dataset.rename_column('prob_desc_input_spec', 'input_specification')
            dataset = dataset.rename_column('prob_desc_sample_outputs', 'sample_outputs')
            dataset = dataset.rename_column('prob_desc_notes', 'notes')
            dataset = dataset.rename_column('prob_desc_output_spec', 'output_specification')
            dataset = dataset.rename_column('prob_desc_description', 'description')
            dataset = dataset.rename_column('hidden_unit_tests', 'human_testcases')
            dataset = dataset.rename_column('pass_rate', 'human_pass_rate')
            dataset = dataset.rename_column('line_coverage', 'human_line_coverage')
            dataset = dataset.rename_column('branch_coverage', 'human_branch_coverage')
            dataset = dataset.map(update_metrics)
        elif load_data_name == 'program_synthesis_data.jsonl':
            code_uid_list = [str(uuid.uuid4()).replace('-', '') for _ in range(len(dataset))]
            dataset = dataset.add_column('code_uid', code_uid_list)
            dataset = dataset.remove_columns('input_from')
            dataset = dataset.remove_columns('output_to')
            dataset = dataset.remove_columns('tokens')
            dataset = dataset.rename_column('input_spec', 'input_specification')
            dataset = dataset.rename_column('output_spec', 'output_specification')
            dataset = dataset.rename_column('ground_truth', 'human_solution')
            dataset = dataset.map(lambda example: {'lang_cluster': lang_cluster_mapping[example['lang_cluster']]})
            dataset = dataset.map(update_testcases)
        elif load_data_name == 'code_translation_data.jsonl':
            code_uid_list = [str(uuid.uuid4()).replace('-', '') for _ in range(len(dataset))]
            dataset = dataset.add_column('code_uid', code_uid_list)
            dataset = dataset.remove_columns('lang')
            dataset = dataset.rename_column('lang_cluster', 'source_lang_cluster')
            dataset = dataset.map(lambda example: {'source_lang_cluster': lang_cluster_mapping[example['source_lang_cluster']]})
            dataset = dataset.map(lambda example: {'target_lang_cluster': lang_cluster_mapping[example['target_lang_cluster']]})
            dataset = dataset.map(update_testcases)
        elif load_data_name == 'code_repair_data.jsonl':
            code_uid_list = [str(uuid.uuid4()).replace('-', '') for _ in range(len(dataset))]
            dataset = dataset.add_column('code_uid', code_uid_list)
            dataset = dataset.remove_columns('submission_id')
            dataset = dataset.remove_columns('tags')
            dataset = dataset.remove_columns('input_from')
            dataset = dataset.remove_columns('output_to')
            dataset = dataset.remove_columns('tokens')
            dataset = dataset.rename_column('input_spec', 'input_specification')
            dataset = dataset.rename_column('output_spec', 'output_specification')
            dataset = dataset.rename_column('exec_outcome', 'execute_outcome')
            dataset = dataset.rename_column('ground_truth', 'human_solution')
            dataset = dataset.map(lambda example: {'lang_cluster': lang_cluster_mapping[example['lang_cluster']]})
            dataset = dataset.map(update_testcases)
        elif load_data_name == 'code_optimization_data.jsonl':
            dataset = dataset.rename_column('mem_baseline_code_uid', 'memory_baseline_code_uid')
            dataset = dataset.rename_column('mem_baseline_code', 'memory_baseline_source_code')
            dataset = dataset.rename_column('mem_baseline_perf', 'memory_baseline_perf')
            dataset = dataset.rename_column('time_baseline_code', 'time_baseline_source_code')
            dataset = dataset.rename_column('task_description', 'description')
            dataset = dataset.rename_column('memory_baseline_perf', 'memory_baseline_performance')
            dataset = dataset.rename_column('time_baseline_perf', 'time_baseline_performance')
            dataset = dataset.map(update_testcases)

        print(dataset)
        dataset.to_json(str(save_data_path), lines=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    main()
