#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry



def _try_register_areanavdatasetv1():
    try:
        from habitat.datasets.roomnav.roomnav_dataset import (  # noqa: F401
            AreaNavDatasetV1,
        )

    except ImportError as e:
        pointnav_import_error = e

        @registry.register_dataset(name="AreaNav-v1")
        class PointnavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise pointnav_import_error
