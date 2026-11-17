"""Explicit hand-test selection. Body/neutral views never request finger edits."""

def finger_test_rotations(pose_name):
    if pose_name != 'hands':
        return ()
    edits = []
    for side in ('Left', 'Right'):
        axis = (0, 0, 1 if side == 'Left' else -1)
        for digit in ('Index', 'Middle', 'Ring', 'Pinky'):
            for joint in (1, 2, 3):
                edits.append((side+'Hand'+digit+str(joint), axis,
                              75 if side == 'Left' else (10 if joint == 1 else 0)))
        edits.append((side+'HandThumb2', axis, 45 if side == 'Left' else 0))
    return tuple(edits)
