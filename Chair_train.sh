DATASET=Synthetic_NeRF_Chair100_100

# Training a Vanilla NeRF as teacher model
python run_nerf_test.py cfgs/paper/pretrain/$DATASET.yaml

# Extracting occupancy grid from teacher model
python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET.yaml

# Distilling teacher into KiloNeRF model
python local_distill.py cfgs/paper/distill/$DATASET.yaml

# Fine-tuning KiloNeRF model
python run_nerf_test.py cfgs/paper/finetune/$DATASET.yaml



DATASET=Synthetic_NeRF_Chair200_200
python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET.yaml
python local_multidistill.py cfgs/paper/distill/$DATASET.yaml
python run_nerf_test_with_depth.py cfgs/paper/finetune/$DATASET.yaml


DATASET=Synthetic_NeRF_Chair400_400
python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET.yaml
python local_multidistill_multiseg.py cfgs/paper/distill/$DATASET.yaml
python run_nerf_test_with_depth.py cfgs/paper/finetune/$DATASET.yaml
