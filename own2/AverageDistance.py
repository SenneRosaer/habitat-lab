import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

if __name__ == '__main__':
    beacon = open("beacon-7-untrimmed.json", "r")
    beacon = json.load(beacon)
    distances = []
    for ep in beacon['episodes']:
        distances.append(ep['info']['geodesic_distance'])
    print(np.average(np.array(distances)))

    sns.distplot(distances)
    plt.show()
