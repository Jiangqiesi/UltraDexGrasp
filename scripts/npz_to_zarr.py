"""Convert per-episode .npz files into a DP3-compatible zarr dataset.

Usage:
    python scripts/npz_to_zarr.py \
        --npz-root outputs/batch_run \
        --out-root datasets \
        --include-rgb

Output layout (one zarr per executed-hand group):
    datasets/
      single_left/dataset.zarr/    # bodex_mode 0 or 3
      single_right/dataset.zarr/   # bodex_mode 1 or 4
      bimanual/dataset.zarr/       # bodex_mode 2

Each zarr has:
    data/
      point_cloud        (T, 2400, 3) float32
      point_cloud_mask   (T, 2400, 1) uint8
      agent_pos          (T, S)       float32
      action             (T, A)       float32
      object_pose        (T, 7)       float32
      rgb_primary_0      (T, H, W, 3) uint8        (optional)
      rgb_primary_1      (T, H, W, 3) uint8        (optional)
    meta/
      episode_ends       (E,) int64
      bodex_mode         (E,) int8
      object_name        (E,) str
      object_scale       (E,) float32
      object_init_pose   (E, 7) float32
      object_final_pose  (E, 7) float32
      success            (E,) bool
      num_steps          (E,) int32
"""
import argparse
import glob
import json
import os

import numpy as np
import zarr


GROUPS = {
    0: 'single_left', 3: 'single_left',
    1: 'single_right', 4: 'single_right',
    2: 'bimanual',
}

DATA_KEYS_REQUIRED = ['point_cloud', 'point_cloud_mask', 'agent_pos', 'action', 'object_pose']
DATA_KEYS_OPTIONAL = ['rgb_primary_0', 'rgb_primary_1']


def load_episode(path):
    npz = np.load(path, allow_pickle=True)
    meta = json.loads(str(npz['meta'].item()))
    arrays = {k: npz[k] for k in npz.files if k != 'meta'}
    return arrays, meta


def write_group(out_dir, episodes, include_rgb):
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    root = zarr.open(out_dir, mode='w')
    data = root.create_group('data')
    meta = root.create_group('meta')

    keys = list(DATA_KEYS_REQUIRED)
    if include_rgb:
        keys += DATA_KEYS_OPTIONAL

    # concat all episodes per key
    concatenated = {}
    for k in keys:
        concatenated[k] = np.concatenate([ep[0][k] for ep in episodes], axis=0)

    episode_ends = np.cumsum([ep[0]['action'].shape[0] for ep in episodes]).astype(np.int64)

    # per-key chunk sizing: small chunks along time, full along feature dims
    for k, arr in concatenated.items():
        chunks = (min(64, arr.shape[0]),) + arr.shape[1:]
        data.create_dataset(k, data=arr, chunks=chunks, dtype=arr.dtype, overwrite=True)

    meta.create_dataset('episode_ends', data=episode_ends, overwrite=True)
    meta.create_dataset('bodex_mode', data=np.array([ep[1]['bodex_mode'] for ep in episodes], dtype=np.int8), overwrite=True)
    meta.create_dataset('object_name', data=np.array([ep[1]['object_name'] for ep in episodes], dtype=object), object_codec=zarr.codecs.VLenUTF8(), overwrite=True)
    meta.create_dataset('object_scale', data=np.array([ep[1]['object_scale'] for ep in episodes], dtype=np.float32), overwrite=True)
    meta.create_dataset('object_init_pose', data=np.array([ep[1]['object_init_pose'] for ep in episodes], dtype=np.float32), overwrite=True)
    meta.create_dataset('object_final_pose', data=np.array([ep[1]['object_final_pose'] for ep in episodes], dtype=np.float32), overwrite=True)
    meta.create_dataset('success', data=np.array([ep[1]['success'] for ep in episodes], dtype=bool), overwrite=True)
    meta.create_dataset('num_steps', data=np.array([ep[1]['num_steps'] for ep in episodes], dtype=np.int32), overwrite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-root', required=True, help='dir scanned recursively for episode_*.npz')
    parser.add_argument('--out-root', required=True, help='dir where {single_left,single_right,bimanual}/dataset.zarr are written')
    parser.add_argument('--include-rgb', action='store_true', help='store RGB streams (large)')
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.npz_root, '**', 'episode_*.npz'), recursive=True))
    if not paths:
        raise SystemExit(f'No episode_*.npz under {args.npz_root}')

    grouped = {g: [] for g in set(GROUPS.values())}
    for p in paths:
        arrays, meta = load_episode(p)
        if not meta.get('success', False):
            continue
        group = GROUPS[meta['bodex_mode']]
        grouped[group].append((arrays, meta))

    for group, episodes in grouped.items():
        if not episodes:
            print(f'[skip] {group}: no episodes')
            continue
        out_dir = os.path.join(args.out_root, group, 'dataset.zarr')
        print(f'[write] {group}: {len(episodes)} episodes -> {out_dir}')
        write_group(out_dir, episodes, args.include_rgb)

    print('done')


if __name__ == '__main__':
    main()
