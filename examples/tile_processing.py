import math
import torch

def group_gaussian_attributes_by_tiles(meta):
    """
    Group the per-Gaussian intersection-row indices by tile index,
    based on the 'isect_ids' output of isect_tiles(). The code
    decodes tile_id from the 64-bit isect_id, using the 
    [camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits)] layout.

    Args:
        meta: The dictionary produced by 'rasterize_gaussian_images(...)'
              plus the call to 'isect_tiles()'. It must contain:
              - 'tile_width', 'tile_height', 'n_cameras' 
              - 'isect_ids' (LongTensor [n_isects]): each isect_id encodes 
                {camera_id, tile_id, depth}.
              - 'flatten_ids' (LongTensor [n_isects]): 
                row indices in [nnz], for accessing attributes
              - 'means2d', 'radii', 'colors', 'depths', 'opacities', etc. 
                which are all shape [nnz, ...] in packed mode.
    
    Returns:
        tile_list: a list of length (#tiles). Each entry is a *list of indices*
                   that belong to that tile. If a tile is empty, it is an empty list.
        
        So if you want the colors of tile i, you do:
          tile_idxs_i = tile_list[i]
          colors_tile_i = meta["colors"][tile_idxs_i]
    """

    tile_width   = meta["tile_width"]   # number of tiles horizontally
    tile_height  = meta["tile_height"]  # number of tiles vertically
    n_cameras    = meta["n_cameras"]
    isect_ids    = meta["isect_ids"]    # [n_isects], int64
    flatten_ids  = meta["flatten_ids"]  # [n_isects], int32 or int64

    # We want (#tiles) = tile_width * tile_height * n_cameras
    num_tiles = tile_width * tile_height * n_cameras

    # Prepare a container: for each tile index, we'll store a list of intersection rows
    tile_list = [[] for _ in range(num_tiles)]

    # We'll decode tile_id and camera_id from 'isect_ids'.
    # The lower 32 bits are 'depth'. The next bits are 'tile_id', then the top bits are 'camera_id'.
    # We must figure out how many bits tile_id uses, how many bits camera_id uses, etc.
    #
    # A typical approach:
    #   depth = (isect_ids & 0xffffffff)
    #   tile_cam = isect_ids >> 32
    #   tile_id = tile_cam & ((1 << tile_bits) - 1)
    #   camera_id = tile_cam >> tile_bits
    #
    # We can compute tile_bits = ceil(log2(tile_width*tile_height)).
    # We can compute cam_bits  = ceil(log2(n_cameras)).
    # But if these are built by isect_tiles(...) on the GPU side, 
    # the code is consistent. We'll replicate that decode logic.

    # 1) how many bits for tile_id
    tile_count = tile_width * tile_height
    tile_bits  = (tile_count - 1).bit_length()  # smallest #bits to store tile_count-1
    # 2) how many bits for camera_id
    cam_bits   = (n_cameras - 1).bit_length() if n_cameras > 1 else 0

    # We'll decode each isect_id => tile_id => store flatten index
    # Note: watch out if isect_tiles used optional camera bits or not. 
    # If your code handles single-camera differently, adapt as needed.

    # Example decode function:
    def decode_isect_id(isect_id: torch.Tensor):
        # isect_id is int64
        depth_mask = 0xffffffff
        depth_val = isect_id & depth_mask  # lower 32 bits
        tile_cam  = isect_id >> 32         # upper bits
        tile_mask = (1 << tile_bits) - 1   # e.g. for tile_id
        tile_id   = tile_cam & tile_mask
        camera_id = tile_cam >> tile_bits  # the top bits after tile_id
        return tile_id, camera_id, depth_val

    with torch.no_grad():
        tile_id_all, camera_id_all, depth_val_all = decode_isect_id(isect_ids)

    # Now we want the global tile index = camera_id * (tile_width * tile_height) + tile_id
    # because we store them in 0..(num_tiles-1).
    # So:
    global_tile_index = camera_id_all * tile_count + tile_id_all  # shape [n_isects]

    # Finally, push each flatten_idx into tile_list[global_tile_index[i]]
    # flatten_ids[i] is the index in meta["colors"], meta["means2d"], etc.
    for i in range(len(isect_ids)):
        t_idx = global_tile_index[i].item()  # tile index
        f_idx = flatten_ids[i].item()        # row index
        tile_list[t_idx].append(f_idx)

    # done
    return tile_list
