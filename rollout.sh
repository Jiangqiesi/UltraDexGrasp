# left whole hand grasp
python rollout.py \
    --hand 0 \
    --object_scale_list '[0.08]'

# right whole hand grasp
python rollout.py \
    --hand 1 \
    --object_scale_list '[0.08]'

# bimanual grasp
python rollout.py \
    --hand 2 \
    --object_scale_list '[0.25]'

# left three-finger tripod
python rollout.py \
    --hand 3 \
    --object_scale_list '[0.04]'

# right three-finger tripod
python rollout.py \
    --hand 4 \
    --object_scale_list '[0.04]'
