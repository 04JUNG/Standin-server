"""Disjoint reporting regions and conservative hand-graph influence mask."""
import numpy as np


REGIONS = ('hands', 'feet', 'head_neck', 'arms', 'legs', 'torso')


def region_labels(names, weights):
    """Dominant bone only: counts are additive, not anatomical segmentation."""
    names = np.asarray(names)
    dominant = names[np.argmax(weights, axis=1)]
    labels = np.full(len(weights), REGIONS.index('torso'), np.int8)
    for i, name in enumerate(dominant):
        for region, tokens in (
            ('hands', ('Hand',)), ('feet', ('Foot', 'Toe')),
            ('head_neck', ('Head', 'Neck')), ('arms', ('Arm',)),
            ('legs', ('UpLeg', 'Leg')),
        ):
            if any(token in name for token in tokens):
                labels[i] = REGIONS.index(region)
                break
    return labels


def hand_graph_mask(names, weights):
    # Includes palm/root-repair support as well as every possible smoothing
    # vertex. The topology guard also preserves their incident faces/neighbors.
    columns = [i for i, name in enumerate(names) if 'Hand' in name]
    return weights[:, columns].sum(axis=1) > .01


def region_counts(labels, ids=None):
    values = labels if ids is None else labels[ids]
    return {name: int(np.count_nonzero(values == i))
            for i, name in enumerate(REGIONS)}
