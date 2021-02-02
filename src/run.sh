python predict.py --gpu 1 --net resnet34s --ghost_cluster 2 \
        --vlad_cluster 8 --loss softmax \
        --resume ../model/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5 \
        --data_path /mnt/Data/public/dataset/VoxCeleb/voxceleb1/vox1_test_wav/wav