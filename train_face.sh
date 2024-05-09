python train_morphable_diffusion.py -b configs/facescape.yaml \
                           --load_entire_model True \
                           -l ./logs_face \
                           -c ./checkpoints_face \
                           --gpus 0,1 \
                           --finetune_from ./ckpt/syncdreamer-pretrain.ckpt \
                        #    --resume