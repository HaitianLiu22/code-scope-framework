import sys

# TODO code_smell_data删除了difficulty字段
# TODO code_test_data删除了difficulty和lang字段
# lyz和wyk的任务没有length
from pprint import pprint

from datasets import load_dataset
from collections import Counter


def code_smell():
    dataset = load_dataset('json', split='train', data_files='raw/code_smell_data.jsonl')
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.remove_columns('difficulty')
    print(dataset)

    dataset.to_json('datacopy/code_smell_data.jsonl', lines=True)


def code_test():
    dataset = load_dataset('json', split='train', data_files='raw/automated_testing_data.jsonl')
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.remove_columns('difficulty')
    dataset = dataset.remove_columns('lang')
    print(dataset)
    print(dataset['lang_cluster'])

    dataset.to_json('datacopy/automated_testing_data.jsonl', lines=True)


def code_summarization():
    dataset = load_dataset('json', split='train', data_files='raw/code_summarization_data.jsonl')
    dataset.cleanup_cache_files()
    print(dataset)

    # dataset = dataset.remove_columns('difficulty')
    # dataset = dataset.remove_columns('lang')
    # print(dataset)
    print(dataset['LOC'])
    #
    # dataset.to_json('datacopy/automated_testing_data.jsonl', lines=True)


def code_optimization():
    dataset = load_dataset('json', split='train', data_files='raw/code_optimization_data.jsonl')
    dataset.cleanup_cache_files()
    print(dataset)

    # dataset = dataset.remove_columns('difficulty')
    # dataset = dataset.remove_columns('lang')
    # print(dataset)
    print(dataset['lang'])
    #
    # dataset.to_json('datacopy/automated_testing_data.jsonl', lines=True)


def program_synthesis():
    dataset = load_dataset('json', split='train', data_files='raw/program_synthesis_data.jsonl')
    dataset.cleanup_cache_files()
    print(dataset)

    # dataset = dataset.remove_columns('difficulty')
    # dataset = dataset.remove_columns('lang')
    # print(dataset)
    print(dataset['input_from'])
    #
    # dataset.to_json('datacopy/automated_testing_data.jsonl', lines=True)


def code_translation():
    dataset = load_dataset('json', split='train', data_files='raw/code_translation_data.jsonl')
    dataset.cleanup_cache_files()
    print(dataset)

    # dataset = dataset.remove_columns('difficulty')
    # dataset = dataset.remove_columns('lang')
    # print(dataset)
    # print(dataset['input_from'])
    #
    # dataset.to_json('datacopy/automated_testing_data.jsonl', lines=True)


def code_repair():  # TODO code repair名字都错了
    dataset = load_dataset('json', split='train', data_files='./datacopy/code_repair_data.jsonl')
    dataset.cleanup_cache_files()
    print(dataset)

    easy = 0
    hard = 0
    difficulty_counts = Counter(dataset['difficulty'])
    for difficulty, count in difficulty_counts.items():
        print(f'{difficulty}: {count}')
        if difficulty >= 800 and difficulty < 1600:
            easy += count
        if difficulty >= 1600 and difficulty < 2800:
            hard += count
    print('Code Repair:')
    print('Total:', len(dataset['difficulty']))
    print('Easy:', easy)
    print('Hard:', hard)
    print(dataset['lang_cluster'])
    print(len(dataset['lang_cluster']))

    # dataset = dataset.remove_columns('difficulty')
    # dataset = dataset.remove_columns('lang')
    # print(dataset)
    # print(dataset['input_from'])
    #
    # dataset.to_json('datacopy/automated_testing_data.jsonl', lines=True)
    print(dataset[0]['source_code'])


def code_testing():
    # gt = load_dataset('json', split='train', data_files='./raw/automated_testing_data.jsonl')
    # hh = load_dataset('json', split='train', data_files='./automated_testing_data.jsonl')
    # print(gt)
    #
    # gt_code_uid = gt['length']
    # hh_code_uid = hh['length']
    # print(gt_code_uid == hh_code_uid)
    # print(gt_code_uid[0])
    # print(hh_code_uid[0])
    # sys.exit()

    gt = load_dataset('json', split='train', data_files='./raw/automated_testing_data.jsonl')['code_uid']
    code_test_data = load_dataset('json', split='train', data_files='./baga_code_test_data.jsonl')

    code_test_data_human_1 = load_dataset('json', split='train', data_files='./baga_code_test_data_human_1.jsonl')
    code_test_data_human_2 = load_dataset('json', split='train', data_files='./baga_code_test_data_human_2.jsonl')
    code_test_data_human_3 = load_dataset('json', split='train', data_files='./baga_code_test_data_human_3.jsonl')
    code_test_data_human_4 = load_dataset('json', split='train', data_files='./baga_code_test_data_human_4.jsonl')
    code_test_data_human_5 = load_dataset('json', split='train', data_files='./baga_code_test_data_human_5.jsonl')
    code_test_eval_human_1 = load_dataset('json', split='train', data_files='./code_test_eval_human_1.jsonl')
    code_test_eval_human_2 = load_dataset('json', split='train', data_files='./code_test_eval_human_2.jsonl')
    code_test_eval_human_3 = load_dataset('json', split='train', data_files='./code_test_eval_human_3.jsonl')
    code_test_eval_human_4 = load_dataset('json', split='train', data_files='./code_test_eval_human_4.jsonl')
    code_test_eval_human_5 = load_dataset('json', split='train', data_files='./code_test_eval_human_5.jsonl')

    code_test_data = code_test_data.add_column('human_sample_testcases_1', code_test_data_human_1['hidden_unit_tests'])
    code_test_data = code_test_data.add_column('human_sample_testcases_2', code_test_data_human_2['hidden_unit_tests'])
    code_test_data = code_test_data.add_column('human_sample_testcases_3', code_test_data_human_3['hidden_unit_tests'])
    code_test_data = code_test_data.add_column('human_sample_testcases_4', code_test_data_human_4['hidden_unit_tests'])
    code_test_data = code_test_data.add_column('human_sample_testcases_5', code_test_data_human_5['hidden_unit_tests'])

    code_test_data = code_test_data.add_column('human_sample_pass_rate_1', code_test_eval_human_1['pass_rate'])
    code_test_data = code_test_data.add_column('human_sample_pass_rate_2', code_test_eval_human_2['pass_rate'])
    code_test_data = code_test_data.add_column('human_sample_pass_rate_3', code_test_eval_human_3['pass_rate'])
    code_test_data = code_test_data.add_column('human_sample_pass_rate_4', code_test_eval_human_4['pass_rate'])
    code_test_data = code_test_data.add_column('human_sample_pass_rate_5', code_test_eval_human_5['pass_rate'])

    code_test_data = code_test_data.add_column('human_sample_line_coverage_1', code_test_eval_human_1['line_coverage'])
    code_test_data = code_test_data.add_column('human_sample_line_coverage_2', code_test_eval_human_2['line_coverage'])
    code_test_data = code_test_data.add_column('human_sample_line_coverage_3', code_test_eval_human_3['line_coverage'])
    code_test_data = code_test_data.add_column('human_sample_line_coverage_4', code_test_eval_human_4['line_coverage'])
    code_test_data = code_test_data.add_column('human_sample_line_coverage_5', code_test_eval_human_5['line_coverage'])

    code_test_data = code_test_data.add_column('human_sample_branch_coverage_1', code_test_eval_human_1['branch_coverage'])
    code_test_data = code_test_data.add_column('human_sample_branch_coverage_2', code_test_eval_human_2['branch_coverage'])
    code_test_data = code_test_data.add_column('human_sample_branch_coverage_3', code_test_eval_human_3['branch_coverage'])
    code_test_data = code_test_data.add_column('human_sample_branch_coverage_4', code_test_eval_human_4['branch_coverage'])
    code_test_data = code_test_data.add_column('human_sample_branch_coverage_5', code_test_eval_human_5['branch_coverage'])

    a = code_test_data['code_uid']
    b = code_test_data_human_1['code_uid']
    c = code_test_data_human_2['code_uid']
    d = code_test_data_human_3['code_uid']
    e = code_test_data_human_4['code_uid']
    f = code_test_data_human_5['code_uid']

    print(gt == a)
    print(gt == b)
    print(gt == c)
    print(gt == d)
    print(gt == e)
    print(gt == f)

    print(gt)
    print(code_test_data)
    print(code_test_data_human_1)

    code_test_data.to_json(str('automated_testing_data.jsonl'), lines=True)


code_testing()
