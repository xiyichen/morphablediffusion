python train_morphable_diffusion.py -b configs/thuman.yaml \
                           --load_entire_model True \
                           -l ./logs_body \
                           -c ./checkpoints_body \
                           --gpus 0,1 \
                           --finetune_from ./ckpt/syncdreamer-pretrain.ckpt \
                        #    --resume