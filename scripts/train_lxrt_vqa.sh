work_dir=.
lxrt_model=$work_dir/checkpoints/pretrained/model
dataroot=$work_dir/data/1013
img_feat_file=$work_dir/data/img_feat_3ee94.h5
output_dir=$work_dir/checkpoints/lxrt_vqa
bert_model=$work_dir/checkpoints/pretrained/bert-base-uncased

python -u $work_dir/lxmert/src/train_vqa.py \
    --train train --valid valid \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA $lxrt_model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 8 \
    --tqdm \
    --dataroot $dataroot \
    --img_feat_file $img_feat_file \
    --output $output_dir \
    --bert_model $bert_model \
    --numWorkers 4
