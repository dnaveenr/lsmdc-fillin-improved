#!/bin/bash
#SBATCH -A dnaveenr
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END

module load u18/cuda/10.2
module load u18/cudnn/7.6.5-cuda-10.2


mkdir /scratch/dnaveenr/

:'
echo "Transfering train_data from ADA to node."
scp ada:/share3/dnaveenr/fillin_data.zip /scratch/dnaveenr/
scp ada:/share3/dnaveenr/i3d_200.zip /scratch/dnaveenr/
scp ada:/share3/dnaveenr/preprocessed_data.tar.gz /scratch/dnaveenr/

echo "Extracting data..."
unzip -q /scratch/dnaveenr/i3d_200.zip -d /scratch/dnaveenr/data
mv /scratch/dnaveenr/data/i3d2 /scratch/dnaveenr/data/i3d	

unzip /scratch/dnaveenr/fillin_data.zip -d /scratch/dnaveenr/data

tar xvzf /scratch/dnaveenr/preprocessed_data.tar.gz -C /scratch/dnaveenr/data/fillin_data
'

eval "$(conda shell.bash hook)"
conda activate base_ds_env

BASE_DIR="/scratch/dnaveenr"

mkdir -p $BASE_DIR/experiments/

echo "Training started..."

#python -m ipdb train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_new_augmented.json                 --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --learning_rate 5e-5  --gender_loss 0.2 --batch_size 64  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3

python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_new_augmented.json                 --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --learning_rate 5e-5  --gender_loss 0.2 --batch_size 64  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3

#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_new_augmented.json                 --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json                 --use_bert_embedding --bert_embedding_dir $BASE_DIR/data/fillin_data/bert_text_gender_embedding/                 --learning_rate 5e-5  --gender_loss 0.2 --batch_size 64  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3

echo "Training finished..."
