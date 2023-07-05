work_path=.
vqa_model=$work_path/lxmert/ckpt/vqa/BEST.pth
vqg_model=$work_path/lxmert/ckpt/vqg/checkpoint-20000                                                                    
bert_model=$work_path/checkpoints/bert-base-uncased
guesser_model=$work_path/checkpoints/guesser1013/checkpoint-66000
img_feat_file=$work_path/data/img_feat_3ee94.h5
golden_img_feat_file=$work_path/data/img_feat_3ee94_golden.h5
dataroot=$work_path/data/1013

lxrt_path=$work_path/lxmert/src
butd_path=$work_path/bottom-up-attention-vqa

python -u $work_path/self_play_lxrt_lxrt.py \
        --lxrt_path $lxrt_path \
        --bert_path $bert_path \
        --vqa_model $vqa_model \
        --vqg_model $vqg_model \
        --bert_model $bert_model \
        --guesser_model $guesser_model \
        --eval_file $dataroot/spot_diff_test.json \
        --img_feat_file $img_feat_file \
        --golden_img_feat_file $golden_img_feat_file \
        --dataroot $dataroot \
        --mode vqg_vqa_guesser\
        --repeat_time 5 \