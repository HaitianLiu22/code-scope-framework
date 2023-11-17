import warnings
import tiktoken

from pathlib import Path
from datasets import load_dataset


def count_tokens(content):
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(content))

    return num_tokens


def accumulate_automated_testing_tokens(example):
    source_code = example['source_code']

    global automated_testing_total_tokens
    automated_testing_total_tokens += count_tokens(source_code)

    return example


def accumulate_code_optimization_tokens(example):
    mem_baseline_code = example['mem_baseline_code']
    time_baseline_code = example['time_baseline_code']

    global code_optimization_total_tokens
    code_optimization_total_tokens += count_tokens(mem_baseline_code)
    code_optimization_total_tokens += count_tokens(time_baseline_code)

    return example


def accumulate_code_repair_tokens(example):
    source_code = example['source_code']

    global code_repair_total_tokens
    code_repair_total_tokens += count_tokens(source_code)

    return example


def accumulate_code_review_tokens(example):
    old_code = example['old_code']

    global code_review_total_tokens
    code_review_total_tokens += count_tokens(old_code)

    return example


def accumulate_code_smell_tokens(example):
    source_code = example['source_code']

    global code_smell_total_tokens
    code_smell_total_tokens += count_tokens(source_code)

    return example


def accumulate_code_summarization_tokens(example):
    code = example['code']

    global code_summarization_total_tokens
    code_summarization_total_tokens += count_tokens(code)

    return example


def accumulate_code_translation_tokens(example):
    source_code = example['source_code']

    global code_translation_total_tokens
    code_translation_total_tokens += count_tokens(source_code)

    return example


def accumulate_program_synthesis_tokens(example):
    ground_truth = example['ground_truth']

    global program_synthesis_total_tokens
    program_synthesis_total_tokens += count_tokens(ground_truth)

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

    for load_data_name in load_data_name_list:
        load_data_path = Path(__file__).parent.parent / Path('datacopy') / Path(load_data_name)
        dataset = load_dataset('json', split='train', data_files=str(load_data_path))
        dataset.cleanup_cache_files()
        # print(dataset)

        if load_data_name == 'automated_testing_data.jsonl':
            dataset = dataset.map(accumulate_automated_testing_tokens)
            print('Automated Testing:')
            print('total samples:', len(dataset))
            print('total tokens:', automated_testing_total_tokens)
            print('average tokens/sample:', automated_testing_total_tokens / len(dataset))
        elif load_data_name == 'code_optimization_data.jsonl':
            dataset = dataset.map(accumulate_code_optimization_tokens)
            print('Code Optimization:')
            print('total samples:', len(dataset) * 2)
            print('total tokens:', code_optimization_total_tokens)
            print('average tokens/sample:', code_optimization_total_tokens / (len(dataset) * 2))
        elif load_data_name == 'code_repair_data.jsonl':
            dataset = dataset.map(accumulate_code_repair_tokens)
            print('Code Repair:')
            print('total samples:', len(dataset))
            print('total tokens:', code_repair_total_tokens)
            print('average tokens/sample:', code_repair_total_tokens / len(dataset))
        elif load_data_name == 'code_review_data.jsonl':
            dataset = dataset.map(accumulate_code_review_tokens)
            print('Code Review:')
            print('total samples:', len(dataset))
            print('total tokens:', code_review_total_tokens)
            print('average tokens/sample:', code_review_total_tokens / len(dataset))
        elif load_data_name == 'code_smell_data.jsonl':
            dataset = dataset.map(accumulate_code_smell_tokens)
            print('Code Smell:')
            print('total samples:', len(dataset))
            print('total tokens:', code_smell_total_tokens)
            print('average tokens/sample:', code_smell_total_tokens / len(dataset))
        elif load_data_name == 'code_summarization_data.jsonl':
            dataset = dataset.map(accumulate_code_summarization_tokens)
            print('Code Summarization:')
            print('total samples:', len(dataset))
            print('total tokens:', code_summarization_total_tokens)
            print('average tokens/sample:', code_summarization_total_tokens / len(dataset))
        elif load_data_name == 'code_translation_data.jsonl':
            dataset = dataset.map(accumulate_code_translation_tokens)
            print('Code Translation:')
            print('total samples:', len(dataset))
            print('total tokens:', code_translation_total_tokens)
            print('average tokens/sample:', code_translation_total_tokens / len(dataset))
        elif load_data_name == 'program_synthesis_data.jsonl':
            dataset = dataset.map(accumulate_program_synthesis_tokens)
            print('Program Synthesis:')
            print('total samples:', len(dataset))
            print('total tokens:', program_synthesis_total_tokens)
            print('average tokens/sample:', program_synthesis_total_tokens / len(dataset))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    automated_testing_total_tokens = 0
    code_optimization_total_tokens = 0
    code_repair_total_tokens = 0
    code_review_total_tokens = 0
    code_smell_total_tokens = 0
    code_summarization_total_tokens = 0
    code_translation_total_tokens = 0
    program_synthesis_total_tokens = 0

    main()
