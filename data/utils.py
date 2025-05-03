"""
Author: Elias Mapendo
Date: May 03, 2025
Description:
data/utils.py utility for collate fn.
"""


def collate_fn(batch):
    return tuple(zip(*batch))