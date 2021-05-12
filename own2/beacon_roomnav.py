import argparse
import gzip
from pathlib import Path
import json
import habitat
from habitat.config.default import _C
from habitat.sims import make_sim
from own2.roomnav_generation import generate_roomnav_episode
ISLAND_RADIUS_LIMIT = 1.5



parser = argparse.ArgumentParser(description='Create room goal navigation task dataset for The Beacon 3D scan data.')
parser.add_argument('--room_annotation', default='own2/annotation_complete.json',help="Json file with room annotations")
# Scene dataset path.
parser.add_argument('--scene_dataset',default='data/scene_datasets/beacon' ,help='Scene dataset path containing the *.glb files.')

# Task complexity.
parser.add_argument('--max-distance', default=30, type=int, help='Maximum shortest path distance in meters.')
parser.add_argument('--max-steps', default=500, type=int, help='Maximum number of episode steps.')

# Dataset split. Default values are based on the MP3D PointNav dataset in Habitat.
parser.add_argument('--train-episodes', default=10000, type=int, help='Number of training episodes per scene.')
parser.add_argument('--valid-episodes', default=250, type=int, help='Number of validation episodes per scene.')
parser.add_argument('--test-episodes', default=56, type=int, help='Number of testing episodes per scene.')

# Output folder.
parser.add_argument('--output', default='./beacon/v0/', help='Dataset root folder.')

# Parse arguments.
args = parser.parse_args()

# Setup output folders.
path = Path(args.output)
path.mkdir(parents=True, exist_ok=False)

(path / 'train' / 'content').mkdir(parents=True, exist_ok=False)
(path / 'val' / 'content').mkdir(parents=True, exist_ok=False)
(path / 'test' / 'content').mkdir(parents=True, exist_ok=False)

# Create splits.
scenes_path = Path(args.scene_dataset)

splits = [('train', args.train_episodes), ('val', args.valid_episodes), ('test', args.test_episodes)]
scenes = ['beacon-6', 'beacon-7']

max_steps = args.max_steps
max_distance = args.max_distance

# Room annotation
annotation_json_path = args.room_annotation
annotation_json = open(annotation_json_path)
annotation_json = json.load(annotation_json)

amount_of_rooms = len(annotation_json["regions"])

for split, size in splits:
    print(f"Creating split: {split}")
    print(size)
    # Create an empty split task data set.

    dataset = habitat.Dataset()
    dataset.episodes = []
    with gzip.open(path / split / f'{split}.json.gz', 'wb') as f:
        f.write(dataset.to_json().encode())

    # Create a task dataset for each scene.
    episodes = []
    for scene in scenes:
        # Setup simulator.
        config = _C.clone()
        config.SIMULATOR.SCENE = str(scenes_path / f"{scene}.glb")
        sim = make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)

        closest_dist_limit: float = 1
        furthest_dist_limit: float = 30
        geodesic_to_euclid_min_ratio: float = 1.1

        if scene == 'beacon-7':
            annotation_json = open(annotation_json_path)
            annotation_json = json.load(annotation_json)
        elif scene == 'beacon-6':
            annotation_json = open('own2/annotation-6.json')
            annotation_json = json.load(annotation_json)

        # Setup episode generator.
        generator = generate_roomnav_episode(
            scene,
            sim,
            annotation_json,
            shortest_path_max_steps=max_steps,
            furthest_dist_limit=max_distance,
            num_episodes=size
        )

        # Create scene dataset.
        for e in generator:
            episodes.append(e)
        sim.close()

        # Store scene dataset.
    dataset = habitat.Dataset()
    dataset.episodes = [e for e in episodes]
    with gzip.open(path / split / 'content' / 'beacon-7-untrimmed.json.gz', 'wb') as f:
        f.write(dataset.to_json().encode())
        print("?")



