"""Strict CMU ASF/AMC reader. No network, retargeting, or review decisions.

ASF axes and AMC DOFs rotate about fixed axes in their written order, as in
https://graphics.cs.cmu.edu/nsp/course/cs229/info/ACCLAIMdef.html .
Only fully specified, degree-based CMU captures with rigid bones are accepted.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str
    vector: np.ndarray
    basis: np.ndarray
    dof: tuple[str, ...]


@dataclass
class Acclaim:
    bones: list[Bone]
    frames: list[dict[str, np.ndarray]]
    scale_cm: float

    @classmethod
    def load(cls, asf: Path, amc: Path):
        sections = {}
        section = None
        for raw in asf.read_text(encoding="utf-8-sig").splitlines():
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if line.startswith(":"):
                section = line.split()[0][1:]
                sections[section] = []
            elif section:
                sections[section].append(line)
        units = dict(line.split(maxsplit=1) for line in sections["units"])
        root = dict(line.split(maxsplit=1) for line in sections["root"])
        if (
            units["angle"] != "deg"
            or root["order"] != "TX TY TZ RX RY RZ"
            or root["axis"] != "XYZ"
        ):
            raise ValueError("unsupported Acclaim units/root order")
        if any(
            float(v) != 0 for k in ("position", "orientation") for v in root[k].split()
        ):
            raise ValueError("nonzero ASF root offsets need a separate adapter")
        scale = 2.54 / float(units["length"])
        records, current = {}, None
        for line in sections["bonedata"]:
            if line == "begin":
                current = {}
            elif line == "end":
                if current["name"] in records:
                    raise ValueError("duplicate ASF bone")
                records[current["name"]] = current
                current = None
            elif current is not None:
                key, *value = line.split()
                if key in ("name", "direction", "length", "axis", "dof"):
                    current[key] = " ".join(value)
        children = {}
        for line in sections["hierarchy"]:
            if line not in ("begin", "end"):
                parent, *names = line.split()
                children[parent] = names
        bones, visited = [], set()

        def visit(parent):
            for name in children.get(parent, []):
                if name in visited:
                    raise ValueError("cyclic or repeated ASF hierarchy")
                visited.add(name)
                row = records[name]
                axis = row["axis"].split()
                dof = tuple(row.get("dof", "").split())
                if (
                    axis[-1] != "XYZ"
                    or any(d not in ("rx", "ry", "rz") for d in dof)
                    or len(set(dof)) != len(dof)
                ):
                    raise ValueError("unsupported Acclaim bone channels")
                vector = (
                    np.array(row["direction"].split(), dtype=float)
                    * float(row["length"])
                    * scale
                )
                basis = Rotation.from_euler(
                    "xyz", list(map(float, axis[:3])), degrees=True
                ).as_matrix()
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError("invalid ASF vector")
                bones.append(Bone(name, parent, vector, basis, dof))
                visit(name)

        visit("root")
        if visited != set(records):
            raise ValueError("disconnected ASF bones")
        text = amc.read_text(encoding="utf-8-sig")
        if ":FULLY-SPECIFIED" not in text or ":DEGREES" not in text:
            raise ValueError("requires fully specified degree AMC")
        frames, frame = [], None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", ":")):
                continue
            if line.isdigit():
                if int(line) != len(frames) + 1:
                    raise ValueError("non-contiguous AMC frames")
                frame = {}
                frames.append(frame)
            else:
                name, *values = line.split()
                if frame is None or name in frame:
                    raise ValueError("invalid AMC frame entry")
                frame[name] = np.array(values, dtype=float)
        expected = {"root": 6, **{b.name: len(b.dof) for b in bones if b.dof}}
        if not frames:
            raise ValueError("empty AMC")
        for frame in frames:
            if set(frame) != set(expected) or any(
                len(frame[k]) != n or not np.isfinite(frame[k]).all()
                for k, n in expected.items()
            ):
                raise ValueError("incomplete or nonfinite AMC frame")
        return cls(bones, frames, scale)

    def forward(self, index):
        """World endpoints and rotations; coordinates are Y-up centimetres."""
        frame = self.frames[index]
        ends = {"root": frame["root"][:3] * self.scale_cm}
        rotations = {
            "root": Rotation.from_euler(
                "xyz", frame["root"][3:], degrees=True
            ).as_matrix()
        }
        for bone in self.bones:
            motion = (
                Rotation.from_euler(
                    "".join(d[1] for d in bone.dof), frame[bone.name], degrees=True
                ).as_matrix()
                if bone.dof
                else np.eye(3)
            )
            rotations[bone.name] = (
                rotations[bone.parent] @ bone.basis @ motion @ bone.basis.T
            )
            ends[bone.name] = ends[bone.parent] + rotations[bone.name] @ bone.vector
        return ends, rotations
