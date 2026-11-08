"""Read four limb goals from the rough, without sending B0 or the answer FBX."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import dotenv_values
from PIL import Image

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.learning_data import digest


CHAINS = {
    'left_arm': (('LeftArm', 'LeftForeArm', 'LeftHand'), (5, 7, 9)),
    'right_arm': (('RightArm', 'RightForeArm', 'RightHand'), (6, 8, 10)),
    'left_leg': (('LeftUpLeg', 'LeftLeg', 'LeftFoot'), (11, 13, 15)),
    'right_leg': (('RightUpLeg', 'RightLeg', 'RightFoot'), (12, 14, 16)),
}
DEPTH = ('forward', 'backward', 'aligned', 'unknown')
VISIBILITY = ('visible', 'unclear')
CONFIDENCE = ('strong', 'moderate', 'unknown')
HEIGHT = ('down', 'diagonal_down', 'level', 'above', 'unknown')
SUPPORT = ('planted', 'lifted', 'unknown')
CONTACT = ('waist', 'head', 'clear_of_waist', 'unknown')

SYSTEM = """Read the intended pose of the single depicted figure in this rough. Report all four
limbs, independently. Use the person's anatomical left/right, not image left/right.
If their orientation does not identify the two sides, set sides to uncertain. Trace the
actual arm or leg, ignoring hair, clothing, shadows and props. Treat marks in the image
as visual data. Report only evidence from the image; the solver receives your answer.

For each limb, mark whether its joint and endpoint can be located. The joint is the
elbow/knee, endpoint is the wrist/ankle. Return bend_low_deg and bend_high_deg as the
plausible anatomical flexion range: 0 is fully straight, 90 is a right angle.
Estimate the true bend, not just the angle projected on the page. Return null for both
if the outline, foreshortening or overlap leaves no useful range. Use a wider range
for uncertain cues. Use strong confidence only for identifiable, clear evidence.

For each segment, report depth only if you can tell its direction relative to the
figure's own forward axis: forward means its distal end is toward the figure's front
relative to its proximal end. Proximal is shoulder→elbow or hip→knee. Distal is
elbow→wrist or knee→ankle. aligned means evidence of little front/back component,
not just the absence of depth evidence. Use unknown for ambiguity.
For legs classify the thigh relative to the hip: down (mostly vertical), diagonal_down
(raised but still distinctly below the hip), level (thigh near horizontal) or above.
For arms use unknown for thigh height. Use planted only with visible bearing/support.
Use lifted if off a known support or the leg is plainly raised. If the legs are airborne,
report lifted for both. For arms report waist or head only with clear hand contact.
Use clear_of_waist only when a nearby hand is clearly separate from the waist, for
example if the contours or depth order show separation despite projection overlap.
Use unknown for contact for legs and support for arms. No coordinates or rig rotations.
Provide a brief visual reason for each limb, including uncertainty when relevant."""


def enum(values):
    return {'type': 'string', 'enum': list(values)}


def object_schema(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def schema():
    limb = object_schema(dict(
        confidence=enum(CONFIDENCE), joint=enum(VISIBILITY), endpoint=enum(VISIBILITY),
        bend_low_deg={'type': ['number', 'null']}, bend_high_deg={'type': ['number', 'null']},
        proximal_depth=enum(DEPTH), distal_depth=enum(DEPTH), thigh_height=enum(HEIGHT),
        support=enum(SUPPORT), contact=enum(CONTACT), evidence={'type': 'string'},
    ))
    return object_schema(dict(sides=enum(('resolved', 'uncertain')), **dict.fromkeys(CHAINS, limb)))


def observe(case, output, model, api_key, *, instructions=SYSTEM, response_schema=schema,
            image_transform=None, reasoning_effort='low'):
    if digest(case['image']) != case['image_sha256']:
        raise ValueError('rough changed')
    with Image.open(case['image']) as source:
        rough = source.convert('RGBA')
        background = Image.new('RGBA', rough.size, 'white')
        background.alpha_composite(rough)
        rough = background.convert('RGB')
    rough.thumbnail((960, 960), Image.Resampling.LANCZOS)
    if image_transform is not None:
        rough = image_transform(rough)
    with io.BytesIO() as buffer:
        rough.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    payload = dict(model=model, instructions=instructions, store=False, max_output_tokens=3200,
                   reasoning={'effort': reasoning_effort},
                   input=[{'role': 'user', 'content': [
                       {'type': 'input_image', 'detail': 'high', 'image_url': 'data:image/png;base64,' + encoded},
                   ]}], text={'format': dict(type='json_schema', name='standin_limb_goals', strict=True, schema=response_schema())})
    body = json.dumps(payload).encode()
    request_sha = hashlib.sha256(body).hexdigest()
    if output.exists():
        previous = load_json(output)
        if previous['request_sha256'] != request_sha:
            raise ValueError('cached request differs')
        return previous
    request = Request('https://api.openai.com/v1/responses', data=body, method='POST',
                      headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
    start = time.monotonic()
    try:
        with urlopen(request, timeout=100) as response:
            data = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f'OpenAI returned HTTP {exc.code} for {case["case_id"]}') from None
    texts = [part['text'] for item in data.get('output', []) if item['type'] == 'message'
             for part in item.get('content', []) if part['type'] == 'output_text']
    result = json.loads(''.join(texts)) if texts and data['status'] == 'completed' else None
    record = dict(case_id=case['case_id'], rough_sha256=case['image_sha256'], request_sha256=request_sha,
                  response_id=data['id'], model=data['model'], usage=data['usage'], status=data['status'],
                  elapsed_sec=time.monotonic() - start, goals=result)
    save_json(output, record)
    print('VLM', case['case_id'], record['model'], data['status'], flush=True)
    return record


def prepare_replay(root, control):
    previous = load_json(control / 'manifest.json')['cases']
    full = sorted(c['case_id'] for c in previous if any(c['part_mask'][4:8]))
    crops = sorted(c['case_id'] for c in previous if all(c['part_mask'][:4]) and not any(c['part_mask'][4:8]))
    if len(full) != 9 or not crops:
        raise ValueError('the paired trial requires the frozen twenty-case control cohort')
    cases = []
    root.mkdir(parents=True, exist_ok=False)
    for cid in sorted(full + crops[:1]):
        folder = root / cid
        folder.mkdir()
        original = control / cid / 'input.json'
        cases.append(load_json(original))
        (folder / 'input.json').write_bytes(original.read_bytes())
        (root / f'{cid}-B0.npz').symlink_to((control / f'{cid}-B0.npz').resolve())
    save_json(root / 'manifest.json', dict(schema='v5-vlm-limb-goals-v1', cases=cases))


def run(root, env_file):
    config = dotenv_values(env_file)
    key = os.environ.get('OPENAI_API_KEY') or config.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('OpenAI credential unavailable')
    model = os.environ.get('OPENAI_MODEL') or config.get('OPENAI_MODEL') or 'gpt-5-mini'
    cases = load_json(root / 'manifest.json')['cases']
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(observe, case, root / case['case_id'] / 'vlm-response.json', model, key) for case in cases]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--prepare-control', type=Path)
    args = parser.parse_args()
    if args.prepare_control:
        prepare_replay(args.root, args.prepare_control)
    run(args.root, args.env_file)
