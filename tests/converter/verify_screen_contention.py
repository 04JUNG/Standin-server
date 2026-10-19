"""Opt-in local real-child preview vs required real-mesh screen benchmark.

Caller must hold the shared heavy-check lock. This never calls a deployed API.
The baseline isolates the old semaphore policy with the same current solver and
assets, not a different geometry implementation. Paths stay out of the report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time

from converter_api.runner import BlenderRunner, RunnerSettings
from converter_api.work_queue import WorkControl, WorkStopped, current_work


def run_case(args, *, legacy):
    started_preview = threading.Event()
    stopped_processes = []
    class Runner(BlenderRunner):
        def _command(self, path):
            job = json.loads(path.read_text())
            if job['preview_model']:
                started_preview.set()
            return super()._command(path)
        def _terminate_process_group(self, process, grace):
            super()._terminate_process_group(process, grace)
            stopped_processes.append(process.poll() is not None)
    root = Path(__file__).resolve().parents[2]
    runner = Runner(RunnerSettings(blender_binary=str(args.blender.resolve()),
        worker_path=root/'converter/worker.py', timeout_seconds=60, terminate_grace_seconds=.2))
    runner.inspect_blender()
    if legacy:
        runner._process_slots = threading.BoundedSemaphore(1)
    bvh = args.bvh.read_bytes()
    def convert(control, character, **options):
        token = current_work.set(control)
        try:
            return runner.convert(bvh_bytes=bvh, character_path=character,
                character_id='body-child' if options.get('preview_model') else 'standin-master-v2',
                character_sha256=hashlib.sha256(character.read_bytes()).hexdigest(),
                conversion_id='local-contention-preview' if options.get('preview_model') else 'local-contention-screen',
                **options)
        finally:
            current_work.reset(token)
    preview_control = WorkControl('preview', budget_seconds=60)
    before_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    with ThreadPoolExecutor(2) as pool:
        preview = pool.submit(convert, preview_control, args.preview_character,
                              output_scope='half', preview_model=True)
        if not started_preview.wait(5):
            preview.cancel(); raise RuntimeError('preview did not start')
        time.sleep(1)  # Child import/bake already running when safety request arrives.
        start = time.monotonic()
        control = WorkControl('screen', budget_seconds=15)
        screen = pool.submit(convert, control, args.character, hand_contact_screen=True)
        try:
            result = screen.result(65)
            receipt = result.report['hand_contact_screen']
            screen_report = dict(status=receipt['status'],
                intersection_pairs=receipt['intersection_pairs'],
                queue_wait_ms=result.queue_wait_ms, execution_ms=result.execution_ms,
                source_bvh_sha256=result.source_bvh_sha256,
                character_sha256=hashlib.sha256(args.character.read_bytes()).hexdigest(),
                artifact_sha256=result.artifact_sha256)
        except WorkStopped as error:
            screen_report = dict(status='unavailable', reason=error.code,
                                 **control.diagnostics())
        screen_report['arrival_to_result_seconds'] = time.monotonic()-start
        try:
            rendered = preview.result(65)
            preview_report = dict(status='complete', bytes=len(rendered.preview_model))
        except WorkStopped as error:
            preview_report = dict(status='stopped', reason=error.code)
        finally:
            preview_control.cancelled.set()
    after_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = dict(policy='legacy_semaphore' if legacy else 'priority_preempt_cancel',
        required=screen_report, optional=preview_report,
        stopped_processes_exited=all(stopped_processes),
        capacity=1, child_cpu_seconds=(after_cpu.ru_utime+after_cpu.ru_stime-
                                      before_cpu.ru_utime-before_cpu.ru_stime))
    if not legacy:
        assert screen_report['status']=='clear_screen', report
        assert preview_report=={'status':'stopped','reason':'PREVIEW_PREEMPTED'}, report
        assert stopped_processes and all(stopped_processes), report
    return report


def main():
    parser=argparse.ArgumentParser()
    for key in ('blender','character','preview-character','bvh','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    reports=[]
    for legacy in (True,False):
        reports.append(run_case(args,legacy=legacy))
        print(json.dumps(reports[-1]),flush=True)
    report=dict(kind='local real child BVH/FBX/Blender contention; not deployed staging/native E2E',
        host_cpu_count=os.cpu_count(), blender='5.2.0 fbe6228777e7',
        bvh_sha256=hashlib.sha256(args.bvh.read_bytes()).hexdigest(),
        preview_character_sha256=hashlib.sha256(args.preview_character.read_bytes()).hexdigest(),
        runs=reports)
    args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
