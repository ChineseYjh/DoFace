python train.py --dataroot ../datasets/eyes/left/raw2shenlanse \
                --name raw2shenlanse-left \
                --n_epochs 500 \
                --n_epochs_decay 500 \
                --save_epoch_freq 25 \
                --model cycle_gan \
                --gpu_ids 1 \
                --display_port 10005