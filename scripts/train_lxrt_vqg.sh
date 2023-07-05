work_dir=.
lxmert_model=$work_dir/checkpoints/pretrained/model
dataroot=$work_dir/data/1013
img_feat_file=$work_dir/data/img_feat_3ee94.h5
output_dir=$work_dir/checkpoints/lxrt_vqg
bert_model=$work_dir/checkpoints/pretrained/bert-base-uncased

python $work_dir/lxmert/src/train_vqg.py \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --lxmert_model $lxmert_model \
    --bert_model $bert_model \
    --dataroot $dataroot \
    --img_feat_file $img_feat_file \
    --output_dir $output_dir \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --eval_all_checkpoints \
    --max_steps 30000 \
    --save_total_limit 30 \
    --num_workers 8 \
    --save_steps 1000 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \

