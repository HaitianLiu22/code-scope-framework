import sys
import json
import warnings
import evaluate
import numpy as np

from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    bleu_metric = evaluate.load('bleu')
    rouge_metric = evaluate.load('rouge')
    bertscore_metric = evaluate.load('bertscore')
    average = 'weighted'
    lang_cluster_list = ['C', 'C#', 'C++', 'Go', 'Java', 'Javascript', 'PHP', 'Python', 'Ruby']
    diff_tag_list = [0, 1, 2]

    load_result_name_list = [
        'code_review_result_codellama.jsonl',
        'code_review_result_gpt3-5.jsonl',
        'code_review_result_gpt4.jsonl',
        'code_review_result_llama2.jsonl',
        'code_review_result_palm2.jsonl',
        'code_review_result_starcoder.jsonl',
        'code_review_result_vicuna.jsonl',
        'code_review_result_wizardcoder.jsonl'
    ]

    model_name_mapping = {
        'codellama': 'Code LLaMA',
        'gpt3-5': 'GPT-3.5',
        'gpt4': 'GPT-4',
        'llama2': 'LLaMA 2',
        'palm2': 'PaLM 2',
        'starcoder': 'StarCoder',
        'vicuna': 'Vicuna',
        'wizardcoder': 'WizardCoder',
    }

    print('Table:')
    score_1_dict = {}
    score_1_dict['code'] = 0
    score_1_dict['data'] = []
    for load_result_name in load_result_name_list:
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))
        print(dataset)

        print('+' + '——' * 25 + '+')
        print(load_result_name.split('_')[-1].split('.')[0] + ':')
        print('+' + '——' * 25 + '+')
        score_1_item = {}
        score_1_item['model'] = model_name_mapping[load_result_name.split('_')[-1].split('.')[0]]
        evaluation_metrics = []

        diff_tag_references = dataset['diff_tag']
        diff_tag_predictions = dataset['predicted_diff_tag']

        accuracy = round(accuracy_score(y_true=diff_tag_references, y_pred=diff_tag_predictions) * 100, 2)
        evaluation_metrics.append(accuracy)
        score_1_item['accuracy'] = str(accuracy)
        print('accuracy score:', accuracy)

        precision = round(precision_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
        evaluation_metrics.append(precision)
        score_1_item['precision'] = str(precision)
        print('average precision score:', precision)

        recall = round(recall_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
        evaluation_metrics.append(recall)
        score_1_item['recall'] = str(recall)
        print('average recall score:', recall)

        f1 = round(f1_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
        evaluation_metrics.append(f1)
        score_1_item['f1'] = str(f1)
        print('average f1 score:', f1)

        filtered_dataset = dataset.filter(lambda example: example['diff_tag'] == 1)
        review_comment_references = filtered_dataset['review_comment']
        review_comment_predictions = filtered_dataset['predicted_review_comment']

        bleu_results = bleu_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
        bleu = round(bleu_results['bleu'] * 100, 2)
        evaluation_metrics.append(bleu)
        score_1_item['bleu'] = str(bleu)
        print('bleu score:', bleu)

        rouge_results = rouge_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
        rouge = round(rouge_results['rougeL'] * 100, 2)
        evaluation_metrics.append(rouge)
        score_1_item['rouge'] = str(rouge)
        print('rouge score:', rouge)

        bertscore_results = bertscore_metric.compute(predictions=review_comment_predictions, references=review_comment_references, lang='en')
        bertscore = round(np.mean(bertscore_results['f1']) * 100, 2)
        evaluation_metrics.append(bertscore)
        score_1_item['bertscore'] = str(bertscore)
        print('bert score:', bertscore)

        print('evaluation metrics:', evaluation_metrics)
        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_1_item['overall'] = str(overall_score)
        print('+' + '-' * 50 + '+')
        print('overall score:', overall_score)
        print('+' + '-' * 50 + '+')

        score_1_dict['data'].append(score_1_item)

    score_1_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_1_dict)
    save_score_1_path = Path(__file__).parent / Path('scores') / Path('code_review_score_1.json')
    with open(str(save_score_1_path), mode='w', encoding='utf-8') as file:
        json.dump(score_1_dict, file, ensure_ascii=False, indent=2)

    print('Appendix Table:')
    score_2_dict = {}
    score_2_dict['code'] = 0
    score_2_dict['data'] = []
    for load_result_name in load_result_name_list:
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))
        print(dataset)

        lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]

        print('+' + '——' * 25 + '+')
        print(load_result_name.split('_')[-1].split('.')[0] + ':')
        print('+' + '——' * 25 + '+')
        score_2_item = {}
        score_2_item['model'] = model_name_mapping[load_result_name.split('_')[-1].split('.')[0]]
        evaluation_metrics = []
        for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
            print('+' + '-' * 50 + '+')
            print(lang_cluster + ':')
            print('+' + '-' * 50 + '+')

            diff_tag_references = lang_cluster_dataset['diff_tag']
            diff_tag_predictions = lang_cluster_dataset['predicted_diff_tag']

            accuracy = round(accuracy_score(y_true=diff_tag_references, y_pred=diff_tag_predictions) * 100, 2)
            evaluation_metrics.append(accuracy)
            score_2_item[f'{lang_cluster.lower()}_accuracy'] = str(accuracy)
            print('accuracy score:', accuracy)

            precision = round(precision_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
            evaluation_metrics.append(precision)
            score_2_item[f'{lang_cluster.lower()}_precision'] = str(precision)
            print('average precision score:', precision)

            recall = round(recall_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
            evaluation_metrics.append(recall)
            score_2_item[f'{lang_cluster.lower()}_recall'] = str(recall)
            print('average recall score:', recall)

            f1 = round(f1_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
            evaluation_metrics.append(f1)
            score_2_item[f'{lang_cluster.lower()}_f1'] = str(f1)
            print('average f1 score:', f1)

            filtered_lang_cluster_dataset = lang_cluster_dataset.filter(lambda example: example['diff_tag'] == 1)
            review_comment_references = filtered_lang_cluster_dataset['review_comment']
            review_comment_predictions = filtered_lang_cluster_dataset['predicted_review_comment']

            bleu_results = bleu_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
            bleu = round(bleu_results['bleu'] * 100, 2)
            evaluation_metrics.append(bleu)
            score_2_item[f'{lang_cluster.lower()}_bleu'] = str(bleu)
            print('bleu score:', bleu)

            rouge_results = rouge_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
            rouge = round(rouge_results['rougeL'] * 100, 2)
            evaluation_metrics.append(rouge)
            score_2_item[f'{lang_cluster.lower()}_rouge'] = str(rouge)
            print('rouge score:', rouge)

            bertscore_results = bertscore_metric.compute(predictions=review_comment_predictions, references=review_comment_references, lang='en')
            bertscore = round(np.mean(bertscore_results['f1']) * 100, 2)
            evaluation_metrics.append(bertscore)
            score_2_item[f'{lang_cluster.lower()}_bertscore'] = str(bertscore)
            print('bert score:', bertscore)

        print('evaluation metrics:', evaluation_metrics)
        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_2_item['overall'] = str(overall_score)
        print('+' + '-' * 50 + '+')
        print('overall score:', overall_score)
        print('+' + '-' * 50 + '+')

        score_2_dict['data'].append(score_2_item)

    score_2_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_2_dict)
    save_score_2_path = Path(__file__).parent / Path('scores') / Path('code_review_score_2.json')
    with open(str(save_score_2_path), mode='w', encoding='utf-8') as file:
        json.dump(score_2_dict, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
