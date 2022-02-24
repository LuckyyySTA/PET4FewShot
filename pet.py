import argparse
import os
import sys
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AdamW,
    BertModel,
    BertForMaskedLM,
    BertTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from data import create_dataloader, transform_fn_dict, convert_example, PetDataset
from evaluate import do_evaluate
def do_train(args):
    set_seed(args.seed)
    label_normalize_json = os.path.join("../label_normalized", args.task_name + '.json')

    # model
    model = BertForMaskedLM.from_pretrained(args.language_model)
    tokenizer = BertTokenizer.from_pretrained(args.language_model)

    # map y
    label_norm_dict = None
    
    with open(label_normalize_json, 'r', encoding="utf-8") as f:
        label_norm_dict = json.load(f)
    
    train_data, public_test_data, test_data = [], [], []
    
    with open("../FewCLUE/datasets/" + args.task_name + '/train_0.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dic = json.loads(line)
            train_data.append(dic)
    
    with open("../FewCLUE/datasets/" + args.task_name + '/test_public.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dic = json.loads(line)
            public_test_data.append(dic)
    
    with open("../FewCLUE/datasets/" + args.task_name + '/test.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dic = json.loads(line)
            test_data.append(dic)

    transform_func = transform_fn_dict[args.task_name]

    transformed_train_data = [transform_func(single_example, label_norm_dict, args.pattern_id) for single_example in train_data]
    transformed_public_test_data = [transform_func(single_example, label_norm_dict, args.pattern_id) for single_example in public_test_data]
    transformed_test_data = [transform_func(single_example, label_norm_dict, True, args.pattern_id) for single_example in test_data]

    convert_train_data = [convert_example(single_example, tokenizer, args.max_seq_length) for single_example in transformed_train_data]
    convert_public_test_data = [convert_example(single_example, tokenizer, args.max_seq_length) for single_example in transformed_public_test_data]
    convert_test_data = [convert_example(single_example, tokenizer, args.max_seq_length, is_test=True) for single_example in transformed_test_data]

    train_ds, public_test_ds, test_ds = PetDataset(convert_train_data), PetDataset(convert_public_test_data), PetDataset(convert_test_data)
    train_data_loader = create_dataloader(train_ds, args.batch_size)
    public_test_data_loader = create_dataloader(public_test_ds, args.batch_size)
    test_data_loader = create_dataloader(test_ds, args.batch_size)
    num_training_steps = len(train_data_loader) * args.epochs
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_proportion * num_training_steps,
        num_training_steps=num_training_steps,
    )

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = torch.load(args.init_from_ckpt)
        model.load_state_dict(state_dict)
        print("warmup from:{}".format(args.init_from_ckpt))
    
    mlm_loss_fn = torch.nn.CrossEntropyLoss()
    evaluate_fn = do_evaluate
    max_test_acc = 0.0
    global_step = 0
    tic_train = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):
            src_ids = batch[0]
            token_type_ids = batch[1]
            masked_positions = batch[2]
            masked_lm_labels = batch[3]

            max_len = src_ids.shape[1]
            new_masked_positions = []

            for bs_index, mask_pos in enumerate(masked_positions.numpy()):
                for pos in mask_pos:
                    new_masked_positions.append(bs_index * max_len + pos)
            new_masked_positions = torch.tensor(np.array(new_masked_positions).astype("int64"))
            output = model(input_ids=src_ids, token_type_ids=token_type_ids)
            output_logits = output.logits
            output_logits = torch.reshape(output_logits, (-1, output_logits.shape[-1]))
            maksed_output = torch.index_select(output_logits, 0, new_masked_positions)
            masked_lm_labels = torch.flatten(masked_lm_labels)
            loss = mlm_loss_fn(maksed_output, masked_lm_labels)
            global_step += 1
            if global_step % 10  == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        save_dir = os.path.join(args.save_dir, "model_%d" %global_step)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_param_path = os.path.join(save_dir, 'model_state.pdparams')
        torch.save(model.state_dict(), save_param_path)
        
        test_accuracy, total_num = evaluate_fn(
            model, tokenizer, public_test_data_loader, label_norm_dict)
        print("epoch:{}, test_accuracy:{:.3f}, total_num:{}".format(
                epoch, test_accuracy, total_num))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        help="The task_name to be evaluated")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--save_dir",
        default='./checkpoint',
        type=str,
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.0,
        type=float,
        help="Linear warmup proption over the training process.")
    parser.add_argument(
        "--pattern_id", default=0, type=int, help="pattern id of pet")
    parser.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None,
        help="The path of checkpoint to be loaded.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="random seeds for initialization")
    parser.add_argument(
        "--output_dir",
        default='./output',
        type=str,
        help="The output directory where to save output")
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument(
        '--save_steps',
        type=int,
        default=10000,
        help="Inteval steps to save checkpoint")
    parser.add_argument("--if_save_checkpoints", action='store_true')
    parser.add_argument(
        "--index",
        required=True,
        type=str,
        default="0",
        help="must be in [0, 1, 2, 3, 4, all]")
    parser.add_argument(
        '--language_model',
        type=str,
        default='bert-base-chinese',
        choices=['bert-base-chinese'],
        help="Language model")
    parser.add_argument(
        "--rdrop_coef", 
        default=0.0, 
        type=float, 
        help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")

    args = parser.parse_args()
    do_train(args)
