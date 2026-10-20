"""Visual checks of input, measured skeleton and unmodified search result."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..review.catalog import Catalog
from ..storage import read_json

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def comparison_sheets(root: Path, ids: list[str], output: Path, evaluation='before.json'):
    inputs = {row['id']: row for row in read_json(root / 'inputs.json')}
    rows = {row['id']: row for row in read_json(root / evaluation)['images']}
    catalog = Catalog(Path('data'), Path('data/curation'))
    poses = {pose.pose_id: pose for pose in catalog.all()}
    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 15)
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(ids), 5):
        page_ids = ids[start:start + 5]
        canvas = Image.new('RGB', (1400, 350 * len(page_ids)), 'white'); draw = ImageDraw.Draw(canvas)
        for index, identity in enumerate(page_ids):
            row = rows.get(identity, {'people': []}); y = index * 350
            with Image.open(inputs[identity]['path']) as source:
                image = Image.new('RGBA', source.size, 'white'); image.alpha_composite(source.convert('RGBA')); image = image.convert('RGB')
            thumb = ImageOps.contain(image, (290, 300)); canvas.paste(thumb, (0, y))
            draw.text((2, y + 305), identity, font=font, fill='black')
            people = sorted(row['people'], key=lambda p: p['mean_body_score'], reverse=True)[:3]
            for person_index, person in enumerate(people):
                x = 300 + person_index * 365
                overlay = image.copy(); pen = ImageDraw.Draw(overlay)
                points, scores = person['keypoints'], person['scores']
                for a, b in EDGES:
                    if min(scores[a], scores[b]) >= .3:
                        pen.line([tuple(points[a]), tuple(points[b])], fill='#d31631', width=max(2, image.width // 180))
                for joint in range(5, 17):
                    if scores[joint] >= .3:
                        xx, yy = points[joint]; radius=max(3,image.width/150)
                        pen.ellipse((xx-radius, yy-radius, xx+radius, yy+radius), fill='#006bb5')
                overlay = ImageOps.contain(overlay, (180, 290)); canvas.paste(overlay, (x, y))
                if person['hits']:
                    hit = person['hits'][0]; pose = poses.get(hit['pose_id'])
                    if pose and hit['view'] in pose.thumbnails and pose.thumbnails[hit['view']].exists():
                        with Image.open(pose.thumbnails[hit['view']]) as source:
                            canvas.paste(source.resize((180,180)), (x + 182, y + 50))
                    draw.text((x, y+294), f"p{person['person']} joints={person['body_visible']} d={hit['distance']:.3f}", font=font, fill='black')
                    draw.text((x,y+317), hit['pose_id'][:39], font=font, fill='black')
        canvas.save(output / f'page-{start//5:02d}.jpg', quality=90)
