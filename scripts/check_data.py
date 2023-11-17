import sys
import warnings

from pathlib import Path
from collections import Counter
from pprint import pprint

import numpy as np
from datasets import load_dataset


def check_code_summarization_data(dataset):
    print(set(dataset['lang']))
    supported_lang_cluster_list = ['F#', 'ARM_Assembly', 'SQL', 'Rust', 'VBScript', 'Java', 'Ruby', 'SAS', 'Delphi', 'COBOL', 'Scala', 'C#', 'Zig', 'C', 'Haskell', 'ABAP', 'MATLAB', 'AWK', 'D',
                                   'PowerShell', 'X86_Assembly', 'Ada', 'Perl', 'C++', 'Go', 'R', 'Fortran', 'Groovy', 'Lua', 'Dart', 'ColdFusion', 'Visual_Basic', 'JavaScript', 'Julia', 'Kotlin',
                                   'Python', 'Visual_Basic_.NET', 'Logo', 'Swift', 'PHP', 'Objective-C', 'Scheme', 'Prolog']
    lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang'] == lang_cluster) for lang_cluster in supported_lang_cluster_list]
    print('Code Summarization:')
    for (supported_lang_cluster, lang_cluster_dataset) in zip(supported_lang_cluster_list, lang_cluster_dataset_list):
        print(f'{supported_lang_cluster} = Total: {len(lang_cluster_dataset)}')


def check_code_smell_tokens(dataset):
    pass


def check_code_review_data(dataset):
    pass


def check_automated_testing_data(example):
    example['human_sample_pass_rate'] = np.mean([example[f'human_sample_pass_rate_{i + 1}'] for i in range(5)])
    example['human_sample_line_coverage'] = np.mean([example[f'human_sample_line_coverage_{i + 1}'] for i in range(5)])
    example['human_sample_branch_coverage'] = np.mean([example[f'human_sample_branch_coverage_{i + 1}'] for i in range(5)])

    return example


def check_program_synthesis_data(dataset):
    print(set(dataset['lang_cluster']))
    supported_lang_cluster_list = ['go', 'perl', 'c', 'c#', 'kotlin', 'ruby', 'javascript', 'python', 'c++', 'rust', 'delphi', 'd', 'php', 'java']
    lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in supported_lang_cluster_list]
    print('Program Synthesis:')
    for (supported_lang_cluster, lang_cluster_dataset) in zip(supported_lang_cluster_list, lang_cluster_dataset_list):
        easy = 0
        hard = 0
        difficulty_counts = Counter(lang_cluster_dataset['difficulty'])
        for difficulty, count in difficulty_counts.items():
            if difficulty >= 800 and difficulty < 1600:
                easy += count
            elif difficulty >= 1600 and difficulty < 2800:
                hard += count
            else:
                print('error')
        print(f'{supported_lang_cluster} = Total: {len(lang_cluster_dataset)} Easy: {easy} Hard: {hard}')


def check_code_translation_data(dataset):
    print(set(dataset['lang']))
    print(set(dataset['lang_cluster']))
    supported_lang_cluster_list = ['go', 'perl', 'c', 'c#', 'kotlin', 'ruby', 'javascript', 'python', 'c++', 'rust', 'delphi', 'd', 'php', 'java']
    lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in supported_lang_cluster_list]
    print('Code Translation:')
    for (supported_lang_cluster, lang_cluster_dataset) in zip(supported_lang_cluster_list, lang_cluster_dataset_list):
        easy = 0
        hard = 0
        difficulty_counts = Counter(lang_cluster_dataset['difficulty'])
        for difficulty, count in difficulty_counts.items():
            if difficulty >= 800 and difficulty < 1600:
                easy += count
            elif difficulty >= 1600 and difficulty < 2800:
                hard += count
            else:
                print('error')
        print(f'{supported_lang_cluster} = Total: {len(lang_cluster_dataset)} Easy: {easy} Hard: {hard}')


def check_code_repair_data(dataset):
    print(set(dataset['lang']))
    print(set(dataset['lang_cluster']))
    supported_lang_cluster_list = ['go', 'perl', 'c', 'c#', 'kotlin', 'ruby', 'javascript', 'python', 'c++', 'rust', 'delphi', 'd', 'php', 'java']
    lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in supported_lang_cluster_list]
    print('Code Repair:')
    for (supported_lang_cluster, lang_cluster_dataset) in zip(supported_lang_cluster_list, lang_cluster_dataset_list):
        easy = 0
        hard = 0
        difficulty_counts = Counter(lang_cluster_dataset['difficulty'])
        for difficulty, count in difficulty_counts.items():
            if difficulty >= 800 and difficulty < 1600:
                easy += count
            elif difficulty >= 1600 and difficulty < 2800:
                hard += count
            else:
                print('error')
        print(f'{supported_lang_cluster} = Total: {len(lang_cluster_dataset)} Easy: {easy} Hard: {hard}')


def check_code_optimization_data(dataset):
    print(set(dataset['lang']))
    supported_lang_cluster_list = ['go', 'perl', 'c', 'c#', 'kotlin', 'ruby', 'javascript', 'python', 'c++', 'rust', 'delphi', 'd', 'php', 'java']
    lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang'] == lang_cluster) for lang_cluster in supported_lang_cluster_list]
    print('Code Translation:')
    for (supported_lang_cluster, lang_cluster_dataset) in zip(supported_lang_cluster_list, lang_cluster_dataset_list):
        easy = 0
        hard = 0
        difficulty_counts = Counter(lang_cluster_dataset['difficulty'])
        for difficulty, count in difficulty_counts.items():
            if difficulty >= 800 and difficulty < 1600:
                easy += count
            elif difficulty >= 1600 and difficulty < 2800:
                hard += count
            else:
                print('error')
        print(f'{supported_lang_cluster} = Total: {len(lang_cluster_dataset)} Easy: {easy} Hard: {hard}')


def main():
    load_data_name_list = [
        # 'code_summarization_data.jsonl',
        # 'code_smell_data.jsonl',
        # 'code_review_data.jsonl',
        # 'automated_testing_data.jsonl',
        'program_synthesis_data.jsonl',
        # 'code_translation_data.jsonl',
        # 'code_repair_data.jsonl',
        # 'code_optimization_data.jsonl'
    ]

    for load_data_name in load_data_name_list:
        load_data_path = Path(__file__).parent.parent / Path('datacopy') / Path(load_data_name)
        dataset = load_dataset('json', split='train', data_files=str(load_data_path))
        dataset.cleanup_cache_files()
        print(dataset)

        if load_data_name == 'automated_testing_data.jsonl':
            # check_automated_testing_data(dataset)
            pprint(dataset[0])
            # print('Automated Testing:')

        elif load_data_name == 'code_optimization_data.jsonl':
            # check_code_optimization_data(dataset)
            print('Code Optimization:')
            pprint(dataset[0])

        elif load_data_name == 'code_repair_data.jsonl':

            # check_code_repair_data(dataset)
            pprint(dataset[0])
            pprint(set(dataset['lang_cluster']))

        elif load_data_name == 'code_review_data.jsonl':
            check_code_review_data(dataset)
            print('Code Review:')

        elif load_data_name == 'code_smell_data.jsonl':
            check_code_smell_tokens(dataset)
            print('Code Smell:')

        elif load_data_name == 'code_summarization_data.jsonl':
            # check_code_summarization_data(dataset)
            pprint(dataset[0])
            print(set(dataset['lang']))

        elif load_data_name == 'code_translation_data.jsonl':
            # check_code_translation_data(dataset)
            pprint(dataset[0])

        elif load_data_name == 'program_synthesis_data.jsonl':
            # check_program_synthesis_data(dataset)

            print(eval(dataset[0]['testcases']))

            testcases = eval(dataset[0]['testcases'])
            print(testcases[0])
            for testcase in testcases:
                del testcase['time']
                del testcase['mem']
            print(testcases[0])


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

    # main()
    hh = load_dataset('json', split='train', data_files='./automated_testing_data.jsonl')
    hh = hh.map(check_automated_testing_data)
    x = hh['human_sample_pass_rate']+hh['human_sample_line_coverage']+hh['human_sample_branch_coverage']
    print(x)
    print(len(x))
    print(np.mean(x))

    print(hh)
