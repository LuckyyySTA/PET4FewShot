python  predict.py \
        --task_name "tnews" \
        --device gpu \
        --init_from_ckpt "./tnews/model_60/model_state.pdparams" \
        --output_dir "./tnews/output" \
        --batch_size 32 \
        --max_seq_length 512
