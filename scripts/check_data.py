import warnings
import numpy as np

from pathlib import Path
from collections import Counter
from datasets import load_dataset


def main():
    load_data_name_list = [
        # 'code_summarization_data.jsonl',
        'code_smell_data.jsonl',
        # 'code_review_data.jsonl',
        # 'automated_testing_data.jsonl',
        # 'program_synthesis_data.jsonl',
        # 'code_translation_data.jsonl',
        # 'code_repair_data.jsonl',
        # 'code_optimization_data.jsonl'
    ]

    for load_data_name in load_data_name_list:
        load_data_path = Path(__file__).parent.parent / Path('data') / Path(load_data_name)
        dataset = load_dataset('json', split='train', data_files=str(load_data_path))
        dataset.cleanup_cache_files()
        print(dataset)

        if load_data_name == 'code_summarization_data.jsonl':
            print('Code Summarization:')
            print('Languages:', set(dataset['lang_cluster']))
            print('#Languages:', len(set(dataset['lang_cluster'])))
            lang_cluster_counts = Counter(dataset['lang_cluster'])
            for lang_cluster, count in lang_cluster_counts.items():
                print(f'#{lang_cluster}: {count}')
        elif load_data_name == 'code_smell_data.jsonl':
            print('Code Smell:')
            print('Languages:', set(dataset['lang_cluster']))
            print('#Languages:', len(set(dataset['lang_cluster'])))
            lang_cluster_counts = Counter(dataset['lang_cluster'])
            for lang_cluster, count in lang_cluster_counts.items():
                print(f'#{lang_cluster}: {count}')
            print('Smells:', set(dataset['smell']))
            smell_counts = Counter(dataset['smell'])
            for smell, count in smell_counts.items():
                print(f'#{smell}: {count}')
            lang_cluster_smell_counts = Counter(list(zip(dataset['lang_cluster'], dataset['smell'])))
            for lang_cluster_smell, count in lang_cluster_smell_counts.items():
                print(f'#{lang_cluster_smell}: {count}')
        elif load_data_name == 'code_review_data.jsonl':
            print('Code Review:')
            print('Languages:', set(dataset['lang_cluster']))
            print('#Languages:', len(set(dataset['lang_cluster'])))
            lang_cluster_counts = Counter(dataset['lang_cluster'])
            for lang_cluster, count in lang_cluster_counts.items():
                print(f'#{lang_cluster}: {count}')
            print('Diff Tags:', set(dataset['diff_tag']))
            diff_tag_counts = Counter(dataset['diff_tag'])
            for diff_tag, count in diff_tag_counts.items():
                print(f'#{diff_tag}: {count}')
            lang_cluster_diff_tag_counts = Counter(list(zip(dataset['lang_cluster'], dataset['diff_tag'])))
            for lang_cluster_diff_tag, count in lang_cluster_diff_tag_counts.items():
                print(f'#{lang_cluster_diff_tag}: {count}')
            print('#Empty review comments:', dataset['review_comment'].count(''))
        elif load_data_name == 'automated_testing_data.jsonl':
            print('Automated Testing:')
            print('Languages:', set(dataset['lang_cluster']))
            print('#Languages:', len(set(dataset['lang_cluster'])))
            lang_cluster_counts = Counter(dataset['lang_cluster'])
            for lang_cluster, count in lang_cluster_counts.items():
                print(f'#{lang_cluster}: {count}')
            lang_cluster_list = ['Python', 'C', 'C++', 'Java']
            lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]
            metrics = []
            for (lang_cluster, lang_cluster_dataset) in zip(lang_cluster_list, lang_cluster_dataset_list):
                print(lang_cluster + ':')
                pass_rate = round(float(np.mean(lang_cluster_dataset['human_sample_pass_rate'])), 2)
                metrics.append(pass_rate)
                print('Average Pass Rate:', pass_rate)
                line_coverage = round(float(np.mean(lang_cluster_dataset['human_sample_line_coverage'])), 2)
                metrics.append(line_coverage)
                print('Average Line Coverage:', line_coverage)
                branch_coverage = round(float(np.mean(lang_cluster_dataset['human_sample_branch_coverage'])), 2)
                metrics.append(branch_coverage)
                print('Average Branch Coverage:', branch_coverage)
            print('Metrics:', metrics)
            overall_score = round(float(np.mean(metrics)), 2)
            print('Overall Score:', overall_score)
        elif load_data_name == 'program_synthesis_data.jsonl':
            print('Program Synthesis:')
            print('Languages:', set(dataset['lang_cluster']))
            print('#Languages:', len(set(dataset['lang_cluster'])))
            lang_cluster_counts = Counter(dataset['lang_cluster'])
            for lang_cluster, count in lang_cluster_counts.items():
                print(f'#{lang_cluster}: {count}')
            lang_cluster_list = ['Python', 'JavaScript', 'D', 'Go', 'Kotlin', 'PHP', 'C', 'Delphi', 'Ruby', 'C#', 'Java', 'Perl', 'C++', 'Rust']
            lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]
            for (lang_cluster, lang_cluster_dataset) in zip(lang_cluster_list, lang_cluster_dataset_list):
                easy = 0
                hard = 0
                difficulty_counts = Counter(lang_cluster_dataset['difficulty'])
                for difficulty, count in difficulty_counts.items():
                    if difficulty >= 800 and difficulty < 1600:
                        easy += count
                    elif difficulty >= 1600 and difficulty < 2800:
                        hard += count
                    else:
                        print('error:', difficulty)
                print(f'{lang_cluster}: Total = {len(lang_cluster_dataset)}, Easy = {easy}, Hard = {hard}')
        elif load_data_name == 'code_translation_data.jsonl':
            print('Code Translation:')
            print('Languages:', set(dataset['source_lang_cluster']))
            print('#Languages:', len(set(dataset['source_lang_cluster'])))
            lang_cluster_counts = Counter(dataset['source_lang_cluster'])
            for lang_cluster, count in lang_cluster_counts.items():
                print(f'#{lang_cluster}: {count}')
            lang_cluster_list = ['Python', 'JavaScript', 'D', 'Go', 'Kotlin', 'PHP', 'C', 'Delphi', 'Ruby', 'C#', 'Java', 'Perl', 'C++', 'Rust']
            lang_cluster_dataset_list = [dataset.filter(lambda example: example['source_lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]
            for (lang_cluster, lang_cluster_dataset) in zip(lang_cluster_list, lang_cluster_dataset_list):
                easy = 0
                hard = 0
                difficulty_counts = Counter(lang_cluster_dataset['difficulty'])
                for difficulty, count in difficulty_counts.items():
                    if difficulty >= 800 and difficulty < 1600:
                        easy += count
                    elif difficulty >= 1600 and difficulty < 2800:
                        hard += count
                    else:
                        print('error:', difficulty)
                print(f'{lang_cluster}: Total = {len(lang_cluster_dataset)}, Easy = {easy}, Hard = {hard}')
        elif load_data_name == 'code_repair_data.jsonl':
            print('Code Repair:')
            print('Languages:', set(dataset['lang_cluster']))
            print('#Languages:', len(set(dataset['lang_cluster'])))
            lang_cluster_counts = Counter(dataset['lang_cluster'])
            for lang_cluster, count in lang_cluster_counts.items():
                print(f'#{lang_cluster}: {count}')
            lang_cluster_list = ['Python', 'JavaScript', 'D', 'Go', 'Kotlin', 'PHP', 'C', 'Delphi', 'Ruby', 'C#', 'Java', 'Perl', 'C++', 'Rust']
            lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]
            for (lang_cluster, lang_cluster_dataset) in zip(lang_cluster_list, lang_cluster_dataset_list):
                easy = 0
                hard = 0
                difficulty_counts = Counter(lang_cluster_dataset['difficulty'])
                for difficulty, count in difficulty_counts.items():
                    if difficulty >= 800 and difficulty < 1600:
                        easy += count
                    elif difficulty >= 1600 and difficulty < 2800:
                        hard += count
                    else:
                        print('error:', difficulty)
                print(f'{lang_cluster}: Total = {len(lang_cluster_dataset)}, Easy = {easy}, Hard = {hard}')
        elif load_data_name == 'code_optimization_data.jsonl':
            print('Code Optimization:')
            print('Languages:', set(dataset['lang']))
            print('#Languages:', len(set(dataset['lang'])))
            lang_cluster_counts = Counter(dataset['lang'])
            for lang_cluster, count in lang_cluster_counts.items():
                print(f'#{lang_cluster}: {count}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    main()
