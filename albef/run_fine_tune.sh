# Remember to change the path in the config files.

python3 -W ignore -m torch.distributed.launch \
  --nproc_per_node=1                \
  --use_env Pretrain.py             \
  --device cuda:0                   \
  --config ./configs/Pretrain.yaml  \
  --checkpoint ./ckpt/albef_4m.pth  \
  --output_dir ./output/Electronics/4m_extend   

python3 -W ignore get_item_embedding.py                                    \
  --device cuda:0                                                          \
  --dataset Electronics                                                    \
  --checkpoint ./output/Electronics/4m_extend/checkpoint_03.pth            \
  --config_train_file ./../gcn/data/Electronics/item_json_for_albef.json 
