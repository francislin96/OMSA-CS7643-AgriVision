CUDA_VISIBLE_DEVICES=1,2,3,0 python train_model_allGPUs.py \
        -config ./config/resnext101_deeplab_config.yaml \
        --dataset_root ./data/images_2021