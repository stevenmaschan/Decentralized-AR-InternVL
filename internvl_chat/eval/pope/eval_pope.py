import argparse
import json
import os


def eval_pope(answers, label_file, question_file=None):
    # Load labels and create mapping by (image, text)
    labels_data = [json.loads(q) for q in open(label_file, 'r')]
    label_map = {}
    for label in labels_data:
        # Use (image, text) as key, removing any prompt text
        label_text = label['text'].strip()
        key = (label['image'], label_text)
        label_map[key] = label['label']
    
    # If question_file is provided, match labels by (image, text) from question file
    # Otherwise, assume labels are in same order as answers (original behavior)
    if question_file:
        questions = {q['question_id']: q for q in [json.loads(line) for line in open(question_file)]}
        label_list = []
        for answer in answers:
            qid = answer['question_id']
            if qid in questions:
                q = questions[qid]
                # Remove prompt part from text for matching
                q_text_clean = q['text'].split('\n')[0].strip() if '\n' in q['text'] else q['text'].strip()
                key = (q['image'], q_text_clean)
                if key in label_map:
                    label_list.append(label_map[key])
                else:
                    # Fallback: try with full text
                    key2 = (q['image'], q['text'].strip())
                    if key2 in label_map:
                        label_list.append(label_map[key2])
                    else:
                        label_list.append(None)
            else:
                label_list.append(None)
    else:
        # Original behavior: assume same order
        label_list = [label['label'] for label in labels_data]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    # Filter out None labels and corresponding answers
    valid_indices = [i for i, label in enumerate(label_list) if label is not None]
    if len(valid_indices) < len(label_list):
        print(f"Warning: {len(label_list) - len(valid_indices)} answers had no matching labels")
        answers = [answers[i] for i in valid_indices]
        label_list = [label_list[i] for i in valid_indices]
    
    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list) if pred_list else 0

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))

    return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-dir', type=str)
    parser.add_argument('--question-file', type=str)
    parser.add_argument('--result-file', type=str)
    args = parser.parse_args()

    f1_list = []
    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = json.loads(open(args.result_file).read())
    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        f1_list.append(eval_pope(cur_answers, os.path.join(args.annotation_dir, file), args.question_file))
        print('====================================')

    print(f'Overall F1: {sum(f1_list)/len(f1_list)*100:.2f}')
