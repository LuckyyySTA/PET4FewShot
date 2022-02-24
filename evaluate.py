import numpy as np
import torch
import torch.nn as nn

@torch.no_grad()
def do_evaluate(model, tokenizer, data_loader, label_normalize_dict):
    
    model.eval()

    total_num = 0
    correct_num = 0

    normed_labels = [
        normalized_label 
        for origin_label, normalized_label in label_normalize_dict.items()
    ]
    label_length = len(normed_labels[0])

    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, masked_lm_labels = batch
        
        max_len = src_ids.shape[1]
        new_masked_positions = []

        for bs_index, mask_pos in enumerate(masked_positions.numpy()):
            for pos in mask_pos:
                new_masked_positions.append(bs_index * max_len + pos)
        new_masked_positions = torch.tensor(
            np.array(new_masked_positions).astype("int32"))

        output = model(input_ids = src_ids, token_type_ids=token_type_ids)
        output_logits = output.logits
        output_logits = torch.reshape(output_logits, (-1, output_logits.shape[-1]))
        prediction_scores = torch.index_select(output_logits, 0, new_masked_positions)
        softmax_fn = torch.nn.Softmax()
        prediction_probs = softmax_fn(prediction_scores)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]
        # prediction_probs: [batch_size, label_length, vocab_size]
        prediction_probs = torch.reshape(prediction_probs, (batch_size, -1, vocab_size)).numpy()
        
        label_ids = np.array([tokenizer(label)["input_ids"][1:-1] for label in normed_labels])
        y_pred = np.ones(shape=[batch_size, len(label_ids)])

        for index in range(label_length):
            y_pred *= prediction_probs[:, index, label_ids[:, index]]

        y_pred_index = np.argmax(y_pred, axis=-1)

        y_true_index=[]
        for masked_lm_label in masked_lm_labels.numpy():
            label_text = "".join(
                tokenizer.convert_ids_to_tokens(list(masked_lm_label)))

            label_index = normed_labels.index(label_text)
            y_true_index.append(label_index)

        y_true_index = np.array(y_true_index)

        total_num += len(y_true_index)
        correct_num += (y_true_index == y_pred_index).sum()

    return 100 * correct_num / total_num, total_num