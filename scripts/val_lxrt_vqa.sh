work_dir=/apdcephfs/share_47076/tmpv_dozheng/Spot_Difference
lxrt_model=$work_dir/checkpoints/pretrained/model
dataroot=$work_dir/data/0206
img_feat_file=$work_dir/data/img_feat_0123.h5
output_dir=$work_dir/checkpoints/lxrt_vqa
bert_model=$work_dir/checkpoints/pretrained/bert-base-uncased
vqa_model=$work_dir/checkpoints/lxrt_vqa/BEST

python -u $work_dir/lxmert/src/train_vqa.py \
    --train train --test val \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA $lxrt_model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 8 \
    --tqdm \
    --load $vqa_model \
    --dataroot $dataroot \
    --img_feat_file $img_feat_file \
    --output $output_dir \
    --bert_model $bert_model \
    --numWorkers 4
