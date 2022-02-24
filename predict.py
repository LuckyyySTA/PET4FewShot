import os
import sys
import random
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from transformers import (
    set_seed,
    BertForMaskedLM,
    BertTokenizer,
    BertModel
)


from data import create_dataloader, transform_fn_dict, convert_example, PetDataset
from evaluate import do_evaluate

predict_file = {
    "tnews": "tnewsf_predict.json"
}

def write_tnews(task_name, output_file, pred_labels):
    test_data, train_few_all_data = [], []

    with open("../FewCLUE/datasets/" + args.task_name + '/test.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dic = json.loads(line)
            test_data.append(dic)
    
    with open("../FewCLUE/datasets/" + args.task_name + '/train_few_all.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dic = json.loads(line)
            train_few_all_data.append(dic)

    def label2id(train_few_all):
        label2id = {}
        for example in train_few_all:
            label = example["label_desc"]
            label_id = example["label"]
            if label not in label2id:
                label2id[label] = str(label_id)
        return label2id
    
    label2id_dict = label2id(train_few_all_data)
    test_example = {}
    with open(output_file, 'w', encoding="utf-8") as f:
        for idx, example in enumerate(test_data):
            print(example)
            test_example["id"] = example["id"]
            test_example["label"] = label2id_dict[pred_labels[idx]]

            str_test_example = json.dumps(test_example) + "\n"
            f.write(str_test_example)


write_fn = {
    "tnews": write_tnews,
}

@torch.no_grad()
def do_predict(model, tokenizer, data_loader, label_normalize_dict):
    model.eval()

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    origin_labels = [
        origin_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    label_length = len(normed_labels[0])

    y_pred_labels = []

    for batch in data_loader:
        src_ids, token_type_ids, masked_positions = batch

        max_len = src_ids.shape[1]
        new_masked_positions = []

        for bs_index, mask_pos in enumerate(masked_positions.numpy()):
            for pos in mask_pos:
                new_masked_positions.append(bs_index * max_len + pos)
        new_masked_positions = torch.tensor(np.array(new_masked_positions).astype('int64'))
        output = model(
            input_ids=src_ids,
            token_type_ids=token_type_ids)
        output_logits = output.logits
        output_logits = torch.reshape(output_logits, (-1, output_logits.shape[-1]))
        prediction_scores = torch.index_select(output_logits, 0, new_masked_positions)

        softmax_fn = torch.nn.Softmax()
        prediction_probs = softmax_fn(prediction_scores)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]

        # prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = torch.reshape(
            prediction_probs, (batch_size, -1, vocab_size)).numpy()

        # [label_num, label_length]
        label_ids = np.array(
            [tokenizer(label)["input_ids"][1:-1] for label in normed_labels])

        y_pred = np.ones(shape=[batch_size, len(label_ids)])

        # Calculate joint distribution of candidate labels
        for index in range(label_length):
            y_pred *= prediction_probs[:, index, label_ids[:, index]]

        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)

        for index in y_pred_index:
            y_pred_labels.append(origin_labels[index])

    return y_pred_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, type=str, help="The task_name to be evaluated")
    parser.add_argument("--p_embedding_num", type=int, default=1, help="number of p-embedding")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--pattern_id", default=0, type=int, help="pattern id of pet")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--output_dir", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to train model, defaults to gpu.")

    args = parser.parse_args()

    set_seed(args.seed)
    label_normalize_json = os.path.join("../label_normalized", args.task_name + ".json")

    # model
    model = BertForMaskedLM.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # map y
    label_norm_dict = None
    with open(label_normalize_json, 'r', encoding="utf-8") as f:
        label_norm_dict = json.load(f)

    # test data
    test_data = []

    with open("../FewCLUE/datasets/" + args.task_name + '/test.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dic = json.loads(line)
            test_data.append(dic)

    # transform & convert example
    transform_func = transform_fn_dict[args.task_name]
    transformed_test_data = [transform_func(single_example, label_norm_dict, True, args.pattern_id) for single_example in test_data]
    convert_test_data = [convert_example(single_example, tokenizer, args.max_seq_length, is_test=True) for single_example in transformed_test_data]

    # data_loader
    test_ds = PetDataset(convert_test_data)
    test_data_loader =create_dataloader(test_ds, args.batch_size)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = torch.load(args.init_from_ckpt)
        model.load_state_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file"
        )
    predict_fn = do_predict

    y_pred_labels = predict_fn(model, tokenizer, test_data_loader, label_norm_dict)
    output_file = os.path.join(args.output_dir, predict_file[args.task_name])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    write_fn[args.task_name](args.task_name, output_file, y_pred_labels)