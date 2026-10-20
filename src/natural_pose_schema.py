"""Untrusted LLM output boundary. No code, joint rotations, or gate overrides."""
from __future__ import annotations

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)


Cm = Annotated[float, Field(ge=-300, le=300)]
Degree = Annotated[float, Field(ge=-180, le=180)]
Vec = Annotated[list[Cm], Field(min_length=3, max_length=3)]
Frame = Literal['w', 'p', 'c', 'h', 'tL', 'tR', 'kL', 'kR']
Hand = Literal['open', 'flat', 'relaxed', 'soft', 'cup', 'grip', 'fist', 'pinch', 'point', 'v', 'thumb', 'ok', 'phone', 'shh']
ObjectSide = Literal['left', 'right', 'both', 'unknown']
ObjectName = Annotated[str, Field(min_length=1, max_length=80)]


class Orientation(Strict):
    pitch: Degree = 0.
    yaw: Degree = 0.
    roll: Degree = 0.


class Root(Orientation):
    x: Cm = 0.
    y: Annotated[float, Field(ge=0, le=250)] = 91.2
    z: Cm = 0.


class Arm(Strict):
    to: Vec
    frame: Frame = 'c'
    pole: Vec | None = None
    palm: Vec | None = None
    pframe: Frame | None = None
    grip: bool = False
    face: Literal['auto', 'down'] | Vec = 'auto'
    pron: Annotated[float, Field(ge=-85, le=85)] = 0.
    flex: Annotated[float, Field(ge=-80, le=70)] = 0.
    dev: Annotated[float, Field(ge=-32, le=22)] = 0.
    load: bool = False


class Foot(Strict):
    mode: Literal['flat', 'free'] = 'flat'
    yaw: Degree = 0.
    pitch: Degree = 0.
    roll: Degree = 0.
    toe: Annotated[float, Field(ge=-70, le=90)] = 0.


class Leg(Strict):
    to: Vec
    frame: Frame = 'w'
    pole: Vec | None = None
    foot: Foot | None = None
    planted: bool | None = None


class Arms(Strict):
    L: Arm | None = None
    R: Arm | None = None


class Legs(Strict):
    L: Leg | None = None
    R: Leg | None = None


class Hands(Strict):
    L: Hand = 'relaxed'
    R: Hand = 'relaxed'


class Prop(Strict):
    kind: Literal['seat', 'desk', 'wall', 'backrest', 'step']
    min: Vec
    max: Vec
    wall: bool = False
    snap: Literal['z-', 'z+', 'x-', 'y+'] | None = None

    @model_validator(mode='after')
    def positive_box(self):
        if any(a >= b for a, b in zip(self.min, self.max)):
            raise ValueError('prop min must be below max on every axis')
        return self


class SpecPatch(Strict):
    root: Root | None = None
    spine: Orientation | None = None
    head: Orientation | None = None
    arms: Arms | None = None
    legs: Legs | None = None
    hands: Hands | None = None
    props: Annotated[list[Prop], Field(max_length=4)] | None = None
    support: Literal['floor', 'seat'] | None = None
    # No support=none, arbitrary support points, collide=false, partner, or joint rotations.


class Tags(Strict):
    posture: Annotated[list[str], Field(min_length=1, max_length=2)]
    action: Annotated[list[str], Field(min_length=1, max_length=8)]
    domain: Annotated[list[str], Field(max_length=3)] = []
    activity: Annotated[list[str], Field(max_length=3)] = []
    object: Annotated[list[str], Field(max_length=5)] = []
    emotion: Annotated[list[str], Field(max_length=3)] = []
    intent: Annotated[list[str], Field(max_length=3)] = []


class BodyParts(Strict):
    lower_posture: Annotated[str, Field(min_length=1, max_length=160)]
    upper_action: Annotated[str, Field(min_length=1, max_length=160)]
    active_hand: Literal['L', 'R', 'both', 'none']
    hands: Hands
    # Authored object-use intent, not a finger preset or a physical-contact receipt.
    # None keeps old stored proposals readable; an explicit {} means no bindings.
    held_object_hands: Annotated[dict[ObjectName, ObjectSide], Field(max_length=5)] | None = None
    head_direction: Annotated[str, Field(min_length=1, max_length=160)]


class PoseProposal(Strict):
    base: Annotated[str, Field(min_length=1, max_length=80)]
    upper: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    spec: SpecPatch
    tags: Tags
    body_parts: BodyParts


def validate_proposal(raw, taxonomy):
    proposal = PoseProposal.model_validate(raw)
    vocabulary = taxonomy['vocabulary']
    for field, values in proposal.tags.model_dump().items():
        if field not in vocabulary or any(v not in vocabulary[field] for v in values):
            raise ValueError('unknown search-v2 tag: ' + field)
    for obj, side in (proposal.body_parts.held_object_hands or {}).items():
        if obj not in proposal.tags.object:
            raise ValueError('held_object_not_in_object_tags')
        # An explicit known empty/pointing hand contradicts a declared grasp.
        # A grasp-compatible shape is necessary here, never sufficient evidence:
        # no binding is ever inferred from the shape or active_hand field.
        sides = ('L', 'R') if side == 'both' else (('L',) if side == 'left' else ('R',) if side == 'right' else ())
        for hand in sides:
            if getattr(proposal.body_parts.hands, hand) not in ('grip', 'fist', 'pinch', 'phone', 'cup'):
                raise ValueError('held_object_hand_shape_incompatible')
    return proposal
