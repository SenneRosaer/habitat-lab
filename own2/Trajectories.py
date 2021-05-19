import cv2
import os
import json
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from collections import Counter
import seaborn as sns
from PIL import Image
from shapely.geometry import LineString, Polygon, Point
import habitat
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps
import imageio
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from matplotlib import image
from matplotlib.path import Path

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def full_trajectory(base):
    matrix = np.zeros((1024, 2757, 3), dtype="uint8")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i, j, 0] = 255
            matrix[i, j, 1] = 255
            matrix[i, j, 2] = 255
    data = None
    files = os.listdir(base)

    for file in files:
        if file[-4:] == ".txt":
            data = None

            with open(base + file, "r") as f:
                data = json.load(f)
                for loc in data['traj']:
                    if matrix[loc[0], loc[1], 1] == 255:
                        for i in range(loc[0] - 10, loc[0] + 10):
                            for j in range(loc[1] - 10, loc[1] + 10):
                                if i > 0 and i < 1024 and j > 0 and j < 2757:
                                    matrix[i, j, 0] = 120
                                    matrix[i, j, 1] = 0
                                    matrix[i, j, 2] = 0

                    else:
                        if matrix[loc[0], loc[1], 1] < 255:
                            for i in range(loc[0] - 5, loc[0] + 5):
                                for j in range(loc[1] - 5, loc[1] + 5):
                                    if i > 0 and i < 1024 and j > 0 and j < 2757:
                                        matrix[i, j, 0] = matrix[i, j, 0] + 10
    image = Image.fromarray(matrix, "RGB")
    image.show()

def start_positions(info):
    matrix = example_get_topdown_map()

    for key in info:
        data = info[key]['content']
        points = [data['traj'][0]]
        for loc in points:
            for i in range(loc[0] - 5, loc[0] + 5):
                for j in range(loc[1] - 5, loc[1] + 5):
                    if i > 0 and i < 1024 and j > 0 and j < 2757:
                        if info[key]['roomsuccess'] == '0.00':
                            matrix[i, j, 0] = 200
                            matrix[i, j, 1] = 0
                            matrix[i, j, 2] = 0
                        else:
                            matrix[i, j, 0] = 0
                            matrix[i, j, 1] = 200
                            matrix[i, j, 2] = 0
    image = Image.fromarray(matrix, "RGB")
    image.show()

def end_positions(info):
    matrix = example_get_topdown_map()

    for key in info:
        data = info[key]['content']
        points = [data['traj'][-1]]
        for loc in points:
            for i in range(loc[0] - 5, loc[0] + 5):
                for j in range(loc[1] - 5, loc[1] + 5):
                    if i > 0 and i < 1024 and j > 0 and j < 2757:
                        if info[key]['roomsuccess'] == '0.00':
                            matrix[i, j, 0] = 200
                            matrix[i, j, 1] = 0
                            matrix[i, j, 2] = 0
                        else:
                            matrix[i, j, 0] = 0
                            matrix[i, j, 1] = 200
                            matrix[i, j, 2] = 0
    Image.fromarray(matrix, "RGB").save('own2/new_images/end_pos.png')


def end_positions_color(info):
    #matrix = example_get_topdown_map()

    image_name = 'own2/colorplan_001.jpg'
    matrix = image.imread(image_name)
    tmp_matrix = np.zeros((len(matrix),len(matrix[0]), 3), dtype="uint8")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            tmp_matrix[i][j][0] = matrix[i][j][0]
            tmp_matrix[i][j][1] = matrix[i][j][1]
            tmp_matrix[i][j][2] = matrix[i][j][2]
    matrix = tmp_matrix

    y_mul = len(matrix)/1024
    x_mul = len(matrix[0]) / 2757
    for key in info:
        data = info[key]['content']
        points = [data['traj'][-1]]
        for loc in points:
            n_loc = [int(loc[0]*x_mul), int(loc[1]*y_mul)]
            for i in range(n_loc[0] - 20, n_loc[0] + 20):
                for j in range(n_loc[1] - 20, n_loc[1] + 20):
                    if i > 0 and i < len(matrix) and j > 0 and j < len(matrix[0]):
                        if info[key]['roomsuccess'] == '0.00':
                            matrix[i,j, 0] = 200
                            matrix[i,j, 1] = 0
                            matrix[i,j, 2] = 0
                        else:
                            matrix[i,j, 0] = 0
                            matrix[i,j, 1] = 200
                            matrix[i,j, 2] = 0

    Image.fromarray(matrix, "RGB").save('own2/new_images/end_pos_color.png')

def both_positions(info):
    matrix = example_get_topdown_map()

    for key in info:
        data = info[key]['content']
        points = [data['traj'][0], data['traj'][-1]]
        for loc in points:
            for i in range(loc[0] - 5, loc[0] + 5):
                for j in range(loc[1] - 5, loc[1] + 5):
                    if i > 0 and i < 1024 and j > 0 and j < 2757:
                        if info[key]['roomsuccess'] == '0.00':
                            matrix[i, j, 0] = 200
                            matrix[i, j, 1] = 0
                            matrix[i, j, 2] = 0
                        else:
                            matrix[i, j, 0] = 0
                            matrix[i, j, 1] = 200
                            matrix[i, j, 2] = 0
    image = Image.fromarray(matrix, "RGB")
    image.show()

def line_between_start_end(info):
    matrix = example_get_topdown_map()

    for key in info:
        data = info[key]['content']
        line = LineString((data['traj'][0], data['traj'][-1]))
        distances = np.arange(0, line.length, 2)
        points = [line.interpolate(distance) for distance in distances] + [line.boundary[1]]
        for p in points:
            loc = [int(p.x), int(p.y)]
            for i in range(loc[0] - 5, loc[0] + 5):
                for j in range(loc[1] - 5, loc[1] + 5):
                    if i > 0 and i < 1024 and j > 0 and j < 2757:
                        if info[key]['roomsuccess'] == '0.00':
                            matrix[i, j, 0] = 200
                            matrix[i, j, 1] = 0
                            matrix[i, j, 2] = 0
                        else:
                            matrix[i, j, 0] = 0
                            matrix[i, j, 1] = 200
                            matrix[i, j, 2] = 0
    image = Image.fromarray(matrix, "RGB")
    image.show()

def example_get_topdown_map():
    config = habitat.get_config(config_paths="configs/tasks/roomnav.yaml")
    dataset = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    map = None
    with habitat.Env(config=config, dataset=dataset) as env:
        env.reset()
        map = maps.get_topdown_map_from_sim(
            env.sim, map_resolution=1024
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        map = recolor_map[map]
    return map

def success_rates(info):
    s_f = dict()

    max_traj = 0
    less_than_max_traj = 0
    for key in info:
        tmp = len(info[key]['content']['traj'])
        if tmp > max_traj:
            max_traj = tmp

    for key in info:
        new_key = info[key]['content']['goal']
        if new_key not in s_f:
            s_f[new_key] = dict()
            s_f[new_key]['success'] = 0
            s_f[new_key]['fail'] = 0
            s_f[new_key]['fail_but_think_success'] = 0

        if info[key]['roomsuccess'] == '1.00':
            s_f[new_key]['success'] += 1
        else:
            if len(info[key]['content']['traj']) <= max_traj - 5:
                s_f[new_key]['fail_but_think_success'] += 1
            else:
                s_f[new_key]['fail'] += 1

    for key in s_f:
        print("Effective "+str(key) +": " + str(s_f[key]['success']/(s_f[key]['success']+s_f[key]['fail']+s_f[key]['fail_but_think_success'])))
        print("Agent thinks " + str(key) +": " + str((s_f[key]['success']+s_f[key]['fail_but_think_success'])/(s_f[key]['success']+s_f[key]['fail']+s_f[key]['fail_but_think_success'])))


def semantic_colors():
    image_name = 'own2/colorplan_cropped2.jpg'
    data = image.imread(image_name)
    print()
    file = open('own2/pixel_annotation.json')
    file_data = json.load(file)["colorplan_cropped2.jpg5267415"]

    tmp = json.load(open('own2/annotation_complete.json'))
    sem = []
    for region in tmp['regions']:
        sem.append(region['semantics'])


    region_colors = {}

    for item in sem:
        if item not in region_colors:
            region_colors[item] = {'red': [], 'green': [], 'blue': []}

    for index,polygon in enumerate(file_data['regions']):
        attributes = polygon["shape_attributes"]
        x = attributes["all_points_x"]
        y = attributes["all_points_y"]
        new_p = []
        for i in range(len(x)):
            new_p.append([x[i],y[i]])
        poly = Polygon(new_p)
        in_p = []
        for i in range(np.min(x), np.max(x)):
            for j in range(np.min(y), np.max(y)):
                p = Point([i,j])
                if p.within(poly):
                    pixel = data[j,i]
                    region_colors[sem[index]]['red'].append(pixel[0])
                    region_colors[sem[index]]['green'].append(pixel[1])
                    region_colors[sem[index]]['blue'].append(pixel[2])


    for key in region_colors:
        sns.distplot(region_colors[key]["red"], hist=False, kde=True, label="red " + key)
        sns.distplot(region_colors[key]["blue"], hist=False, kde=True, label="blue " + key)
        sns.distplot(region_colors[key]["green"], hist=False, kde=True, label="green " + key)

        plt.legend()
        plt.show()

if __name__ == '__main__':
    base = 'video_dir/beacon-6-office/videos/'

    files = os.listdir(base)

    file_information = {}
    for file in files:
        info = file[:-4].split('-')
        file_information[info[0] + '-' + info[1]] = dict()
        for item in info:
            key, value = item.split('=')
            file_information[info[0] + '-' +info[1]][key] = value
        with open(base + file, "r") as f:
            data = json.load(f)
            file_information[info[0] + '-' + info[1]]['content'] = data

    #semantic_colors()
    end_positions(file_information)
    end_positions_color(file_information)
