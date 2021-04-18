import os
import numpy as np
import argparse
import re

from PIL import Image


def get_image_layers(img_dir):
    """Read and validate image layers from file.

    :param img_dir: Location of layer images. File names include layer numbers.
    :return: A list of layers as arrays in sorted order
    """
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
    layers = [np.asarray(Image.open(os.path.join(img_dir, f)))
              for f in sorted(files, key=img_nums.get)]
    if len(set([layer.shape for layer in layers])) != 1:
        raise Exception("Image size mismatch")
    return layers


def dim_scales(n, scales):
    """Determine the "scale" or rate of movement along a dimension (x or y)
    for each layer. `scales` is either empty, of length 1, or it specifies
    all the layer scales. If scales is empty, use 0 for every layer.
    If scales contains a single value, use that value for every layer.

    :param n: The number of layers.
    :param scales: A list of length 0, 1 or n, of floats in [0, 1].
    :return: A list of length n of floats in [0, 1].
    """
    if scales:
        if len(scales) == 1:
            val = scales[0]
            if val > 1 or val < 0:
                raise Exception("Invalid value for scales")
            return [scales[0] for _ in range(n)]
        elif len(scales) == n:
            if any([val > 1 or val < 0 for val in scales]):
                raise Exception("Invalid values for scales")
            return scales
        else:
            raise Exception("Expected {} scale values, got {}".format(n, len(scales)))
    return [0 for _ in range(n)]    # no movement along this dimension


def get_paths(window_size, img_size, row_scales, col_scales, num_frames):
    """Each frame of the gif is made up of windows into each original image layer.
    The function calculates where the corners of these windows are, for each
    layer of each frame.

    :param window_size: A float in (0, 1] representing how large the windows should be,
    relative to the image size
    :param img_size: The shape of the layer images
    :param row_scales: The rates of motion to use in the row dimension
    :param col_scales: The rates of motion to use in the column dimension
    :param num_frames: The number of frames for the resulting gif
    :return: A list of paths, one for each layer, where a path is a length `num_frames`
    that contains corner positions for windows into our layers
    """
    if window_size <= 0 or window_size > 1:
        raise Exception("Invalid window size")
    window_shape = np.array(np.array(img_size) * window_size, dtype=int)
    paths = []

    num_img_rows, num_img_cols, _ = img_size
    num_window_rows, num_window_cols, _ = window_shape

    top_window_row = num_window_rows // 2
    bottom_window_row = num_img_rows - (num_window_rows // 2)
    row_amplitude = (bottom_window_row - top_window_row) // 2

    left_window_col = num_window_cols // 2
    right_window_col = num_img_cols - (num_window_cols // 2)
    col_amplitude = (right_window_col - left_window_col) // 2

    for row_scale, col_scale in zip(row_scales, col_scales):
        row_amplitude *= row_scale
        col_amplitude *= col_scale
        amplitude = np.array([row_amplitude, col_amplitude])

        origin = img_size[:-1] // 2
        path = []
        for t in range(num_frames):
            theta = t * (2 * np.pi / num_frames)
            window_cent = origin + (amplitude * np.array([np.cos(theta), np.sin(theta)]))
            tl_row, tl_col = window_cent - (np.array([num_window_rows, num_window_cols]) // 2)
            br_row, br_col = window_cent + (np.array([num_window_rows, num_window_cols]) // 2)
            path.append([int(x) for x in [tl_row, br_row, tl_col, br_col]])
        paths.append(path)
    return paths


def crop_layer(layer, path):
    """Crop an image layer for each set of corners in `path` to generate
    the final gif frames for that layer.
    """
    ret = []
    for tl_row, br_row, tl_col, br_col in path:
        cropped_img = layer[tl_row:br_row, tl_col:br_col, :]
        ret.append(Image.fromarray(cropped_img))
    return ret


def make_frame(layers):
    """Compose all the layers for a frame."""
    img = layers[0]
    for layer in layers[1:]:
        img.paste(layer, mask=layer)
    return img


def main(args):
    layers = get_image_layers(args.img_dir)
    img_size = np.array(layers[0].shape)
    window_size = args.window_size

    col_scales = dim_scales(len(layers), args.x_scales)
    row_scales = dim_scales(len(layers), args.y_scales)

    paths = get_paths(window_size, img_size, row_scales, col_scales, args.num_frames)
    all_layers = [crop_layer(layer, path) for layer, path in zip(layers, paths)]
    frames = [make_frame(frame_layers) for frame_layers in zip(*all_layers)]
    frames[0].save(args.out_file, save_all=True, append_images=frames[1:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='para some llax!')
    parser.add_argument('img_dir', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('num_frames', type=int)
    parser.add_argument('--window_size', type=float, default=0.9)
    parser.add_argument('--x_scales', nargs='*', type=float, required=False)
    parser.add_argument('--y_scales', nargs='*', type=float, required=False)

    args = parser.parse_args()
    main(args)

