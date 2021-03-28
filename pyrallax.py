import os
import numpy as np
import argparse
import re

from PIL import Image

def get_image_layers(img_dir):
    img_nums = {}
    files = os.listdir(img_dir)
    for f in files:
        if f.startswith('.'):
            continue
        num = re.findall(r'\d+', f)
        if len(num) != 1:
            raise Exception("Multiple/no numbers in filename")
        img_nums[f] = int(num[0])
    files = filter(img_nums.get, files)
    return [np.asarray(Image.open(os.path.join(img_dir, f))) for f in sorted(files, key=img_nums.get)]

def dim_scales(n, scales):
    if scales:
        if len(scales) == 1:
            return [scales[0] for _ in range(n)]
        elif len(scales) == n:
            if any([val > 1 or val < 0 for val in scales]):
                raise Exception("Invalid scales")
            return scales
        else:
            raise Exception("Expected {} scale values, got {}".format(n, len(scales)))
    return [0 for _ in range(n)]

def get_scales(n, x_scales, y_scales):
    col_scales = dim_scales(n, x_scales)
    row_scales = dim_scales(n, y_scales)
    return row_scales, col_scales

def get_paths(window_size, img_size, row_scales, col_scales, num_points):
    paths = []
    for row_scale, col_scale in zip(row_scales, col_scales):
        num_img_rows, num_img_cols, _ = img_size
        num_window_rows, num_window_cols, _ = window_size

        top_window_row = num_window_rows // 2
        bottom_window_row = num_img_rows - (num_window_rows // 2)
        row_amplitude = (bottom_window_row - top_window_row) // 2
        row_amplitude *= row_scale

        left_window_col = num_window_cols // 2
        right_window_col = num_img_cols - (num_window_cols // 2)
        col_amplitude = (right_window_col - left_window_col) // 2
        col_amplitude *= col_scale

        amplitude = np.array([row_amplitude, col_amplitude])

        origin = img_size[:-1] // 2
        path = []
        for t in range(num_points):
            theta = t * (2 * np.pi / num_points)
            window_cent = origin + (amplitude * np.array([np.cos(theta), np.sin(theta)]))
            tl_row, tl_col = window_cent - (np.array([num_window_rows, num_window_cols]) // 2)
            br_row, br_col = window_cent + (np.array([num_window_rows, num_window_cols]) // 2)
            path.append([int(x) for x in [tl_row, br_row, tl_col, br_col]])
        paths.append(path)
    return paths

def rotate_layer(layer, path):
    ret = []
    for tl_row, br_row, tl_col, br_col in path:
        cropped_img = layer[tl_row:br_row, tl_col:br_col, :]
        ret.append(Image.fromarray(cropped_img))
    return ret

def make_frame(layers):
    img = layers[0]
    for layer in layers[1:]:
        img.paste(layer, mask=layer)
    return img

def giphity(out, frames):
    frames[0].save(out, save_all=True, append_images=frames[1:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='para some llax!')
    parser.add_argument('img_dir', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('num_points', type=int)
    parser.add_argument('--x_scales', nargs='*', type=float, required=False)
    parser.add_argument('--y_scales', nargs='*', type=float, required=False)

    args = parser.parse_args()
    layers = get_image_layers(args.img_dir)
    img_size = np.array(layers[0].shape)
    window_size = np.array(np.array(img_size) * .9, dtype=int)

    row_scales, col_scales = get_scales(len(layers), args.x_scales, args.y_scales)
    paths = get_paths(window_size, img_size, row_scales, col_scales, args.num_points)   # assume all the layers are the same shape
    all_layers = [rotate_layer(layer, path) for layer, path in zip(layers, paths)]
    frames = [make_frame(frame_layers) for frame_layers in zip(*all_layers)]
    giphity(args.out_file, frames)
