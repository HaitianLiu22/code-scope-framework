import warnings
import numpy as np

from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    average = 'weighted'
    lang_cluster_list = ['C#', 'Java']
    smell_list = ['large class', 'long method', 'data class', 'blob', 'feature envy', '']

    load_result_name_list = [
        'code_smell_result_codellama.jsonl',
        'code_smell_result_gpt3-5.jsonl',
        'code_smell_result_gpt4.jsonl',
        'code_smell_result_llama2.jsonl',
        'code_smell_result_palm2.jsonl',
        'code_smell_result_starcoder.jsonl',
        'code_smell_result_vicuna.jsonl',
        'code_smell_result_wizardcoder.jsonl'
    ]

    for load_result_name in load_result_name_list:
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))
        print(dataset)

        lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]

        print('+' + '——' * 25 + '+')
        print(load_result_name.split('_')[-1].split('.')[0] + ':')
        print('+' + '——' * 25 + '+')
        evaluation_metrics = []
        for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
            print('+' + '-' * 50 + '+')
            print(lang_cluster + ':')
            print('+' + '-' * 50 + '+')

            references = lang_cluster_dataset['smell']
            predictions = lang_cluster_dataset['predicted_smell']

            accuracy = round(accuracy_score(y_true=references, y_pred=predictions) * 100, 2)
            evaluation_metrics.append(accuracy)
            print('accuracy score:', accuracy)

            precision = round(precision_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
            evaluation_metrics.append(precision)
            print('average precision score:', precision)

            recall = round(recall_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
            evaluation_metrics.append(recall)
            print('average recall score:', recall)

            f1 = round(f1_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
            evaluation_metrics.append(f1)
            print('average f1 score:', f1)

        print('evaluation metrics:', evaluation_metrics)
        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        print('+' + '-' * 50 + '+')
        print('overall score:', overall_score)
        print('+' + '-' * 50 + '+')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
