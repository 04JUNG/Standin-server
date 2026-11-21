"""Run the cached object detector on local images without an external API."""

import json
import os
from pathlib import Path
import subprocess
import sys

class LocalPropDetector:
    def __init__(self, python, model):
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', OPENBLAS_NUM_THREADS='1')
        self.process = subprocess.Popen(
            [str(python), '-u', __file__, str(model)], text=True, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, env=env)

    def observe(self, path):
        self.process.stdin.write(json.dumps(str(path)) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Local object detector exited during inference')
        response = json.loads(line)
        if response.get('error'):
            raise RuntimeError(response['error'])
        return response['hits']

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


def serve(model_dir):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.disable_custom_kernels = True
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir, config=config, local_files_only=True, use_safetensors=True).eval()
    for line in sys.stdin:
        try:
            with Image.open(Path(json.loads(line))) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
            image.thumbnail((720, 720))
            inputs = processor(images=image, text='smartphone. umbrella. sports ball.', return_tensors='pt')
            with torch.inference_mode():
                output = model(**inputs)
            scored = processor.post_process_grounded_object_detection(
                output, input_ids=inputs.input_ids, threshold=.15, text_threshold=.24,
                target_sizes=[(image.height, image.width)])[0]
            labels = scored.get('text_labels')
            if labels is None:
                labels = scored.get('labels', [])
            hits = [dict(label=str(label), score=float(score), box=[float(x) for x in box])
                    for label, score, box in zip(labels, scored['scores'], scored['boxes']) if label]
            print(json.dumps({'hits': hits}), flush=True)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            print(json.dumps({'error': type(error).__name__}), flush=True)


if __name__ == '__main__':
    serve(sys.argv[1])
