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




def end_positions(info, name):
    '''
    Plot all the end positions on an image of the floor
    Args:
        info: The trajectories that will be taken into account
        name: Name of the output image

    Returns:

    '''
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
    Image.fromarray(matrix, "RGB").save('own2/new_images/end_pos'+name+'.png')


def end_positions_color(info, name):
    '''
        Plot all the end positions on a colored image of the floor
        Args:
            info: The trajectories that will be taken into account
            name: Name of the output image

        Returns:

        '''

    #image on which positions are added
    image_name = 'own2/floor7.jpg'
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
        if data['goal'] == 1:
            points = [data['traj'][-1]]
            for loc in points:
                n_loc = [int(loc[0]*x_mul), int(loc[1]*y_mul)]
                for i in range(n_loc[0] - 20, n_loc[0] + 20):
                    for j in range(n_loc[1] - 20, n_loc[1] + 20):
                        if i > 0 and i < len(matrix) and j > 0 and j < len(matrix[0]):
                            if info[key]['roomsuccess'] == '0.00' and len(points) < 490:
                                matrix[i,j, 0] = 200
                                matrix[i,j, 1] = 0
                                matrix[i,j, 2] = 0
                            elif info[key]['roomsuccess'] == '1.00':
                                matrix[i,j, 0] = 0
                                matrix[i,j, 1] = 200
                                matrix[i,j, 2] = 0

    Image.fromarray(matrix, "RGB").save('own2/new_images/end_pos_color'+name+'.png')


def example_get_topdown_map():
    '''
    Generate navigatable top-down map based on the scene specified in a config file
    Returns:

    '''
    config = habitat.get_config(config_paths="configs/tasks/roomnav.yaml")

    map = None
    with habitat.Env(config=config) as env:
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
    '''
    Based on the trajectories plot the percentage of assumed and effective success
    Args:
        info:

    Returns:

    '''
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
        print("Samples: " + str(s_f[key]['success']+s_f[key]['fail']+s_f[key]['fail_but_think_success']))
        print("------------------")

def semantic_colors():
    '''
    Plot RGB distributions for every type based on the annotation and a color ground plan
    Returns:

    '''
    #Image used to obtain colors
    image_name = 'own2/floor7.jpg'
    data = image.imread(image_name)

    #Annotation
    file = open('own2/pixel_annotation_floor7.json')
    file_data = json.load(file)["colorplan_cropped2.jpg5267415"]

    tmp = json.load(open('own2/annotation-7.json'))
    sem = []
    for region in tmp['regions']:
        sem.append(region['semantics'])


    region_colors = {}

    for item in sem:
        if item not in region_colors:
            region_colors[item] = {'red': [], 'green': [], 'blue': []}

    for index,polygon in enumerate(file_data['regions']):
        if index % 10 == 0:
            print("Done " + str(index))
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
        sns.distplot(region_colors[key]["red"], hist=False, kde=True, label="red " + key + " floor 7", color="red")
        sns.distplot(region_colors[key]["blue"], hist=False, kde=True, label="blue " + key + " floor 7", color="blue")
        sns.distplot(region_colors[key]["green"], hist=False, kde=True, label="green " + key + " floor 7", color="green")

        plt.legend()
        plt.show()

if __name__ == '__main__':
    #Directory with files
    base = 'video_dir/beacon-6-all-types/videos/'

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
    name = 'meeting6'
    #
    # success_rates(file_information)
    # end_positions(file_information, name)
    end_positions_color(file_information, name)
