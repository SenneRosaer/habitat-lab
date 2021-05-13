from PIL import Image
import os
import random
import sys

import numpy as np

# %matplotlib inline
from matplotlib import pyplot as plt
import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat import make_dataset

def get_replica_config(config):
    r""" Updates the config to replicate scenes.
    Updates the config to include replicated scenes to support multiple workers
    when there are insufficient workers.
    :param config: Habitat.config
    :return: Habitat.config
    """
    # Determine the number of scenes in this dataset.
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if len(scenes) >= config.NUM_ENVIRONMENTS:
        # Fallback to default handling in habitat.
        return config

    assert config.NUM_ENVIRONMENTS % len(scenes) == 0, "Number of processes must be divisible by number of scenes."

    # Determine number of replicas.
    num_replicas = config.NUM_ENVIRONMENTS // len(scenes)

    # Update config.
    config.defrost()
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = scenes * num_replicas
    config.freeze()

    return config

if __name__ == "__main__":
    # config = get_baselines_config(
    #     "./habitat_baselines/config/pointnav/ppo_pointnav_example.yaml"
    # )
    config = get_baselines_config(
        "./habitat_baselines/config/roomnav/ppo_roomnav.yaml"
    )
    config = get_replica_config(config)
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    trainer = trainer_init(config)
    trainer.train()
    #trainer.eval()

