#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm

from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
import math
from shapely.geometry import Polygon
cv2 = try_cv2_import()
import matplotlib.pyplot as plt

def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    r"""Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own2 risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()


def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view_l: List[np.ndarray] = []
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()
        tmp = observation['sal'].detach().numpy()
        tmp.shape = (256, 256)
        t = plt.cm.hot(tmp)
        out = np.array([np.delete(t[i], np.s_[3:], 1) for i in range(len(t))])
        out = out * 255
        egocentric_view_l.append(out)
        egocentric_view_l.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view_l.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view_l.append(rgb)

    assert (
        len(egocentric_view_l) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view_l, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], egocentric_view.shape[0]
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)


    if 'room' in info:
        if info['scene'] == 7:
            lo = [-8.416683, -4.635123, -11.14608]
            hi = [55.16621, 6.341098, 12.464998]
        else:
            lo = [-29.218403, -1.854079, -11.744533]
            hi = [34.50847, 9.30966, 11.977054]
        grid = (
            abs(hi[2] - lo[2]) / len(top_down_map),
            abs(hi[0] - lo[0]) / len(top_down_map[0]),

        )
        for room in info['room']:
            points = []
            poly = Polygon(room)
            for i in range(-1, len(poly.exterior.coords.xy[0]) - 1):
                p1 = [poly.exterior.coords.xy[0][i],
                      poly.exterior.coords.xy[1][i]]
                p2 = [poly.exterior.coords.xy[0][i + 1],
                      poly.exterior.coords.xy[1][i + 1]]
                if p1 != p2:
                    v = [p2[0] - p1[0], p2[1] - p1[1]]
                    v2 = math.sqrt(
                        math.pow(v[0], 2) + math.pow(v[1], 2))
                    v3 = [v[0] / v2, v[1] / v2]

                    distances = []
                    d = 0
                    while True:
                        d += 0.05
                        if d > v2:
                            break
                        distances.append(d)
                    for i in distances:
                        vt = [v3[0] * i, v3[1] * i]
                        points.append([p1[0] + vt[0], p1[1] + vt[1]])
            for point in points:
                t1 = (math.floor((point[0] - lo[0])/grid[1]), math.floor((point[1] - lo[2])/grid[0]))
                for i in range(t1[1]-1, t1[1]+1):
                    for j in range(t1[0]-1, t1[0]+1):
                        if i < 256:
                            top_down_map[i][j] = np.array([255,127,80])
            frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    if 'roompoints' in info:
        if info['scene'] == 7:
            lo = [-8.416683, -4.635123, -11.14608]
            hi = [55.16621, 6.341098, 12.464998]
        else:
            lo = [-29.218403, -1.854079, -11.744533]
            hi = [34.50847, 9.30966, 11.977054]

        grid = (
            abs(hi[2] - lo[2]) / len(top_down_map),
            abs(hi[0] - lo[0]) / len(top_down_map[0]),

        )
        for room in info['roompoints']:
            points = room
            for point in points:
                t1 = (math.floor((point[0] - lo[0])/grid[1]), math.floor((point[2] - lo[2])/grid[0]))
                for i in range(t1[1]-2, t1[1]+2):
                    for j in range(t1[0]-2, t1[0]+2):
                        if i < 256:
                            top_down_map[i][j] = np.array([80,127,255])
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)

    # if 'point' in info:
    #     t1 = (math.floor((info['point'][0] - lo[0]) / length),
    #           math.floor((-(-info['point'][1] - hi[2])) / length))
    #     for i in range(t1[1] - 1, t1[1] + 1):
    #         for j in range(t1[0] - 1, t1[0] + 1):
    #             top_down_map[i][j] = np.array([255, 0, 0])
    #     frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    return frame


def append_text_to_image(image: np.ndarray, text: str):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final
