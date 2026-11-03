import copy
from pathlib import Path
import pytest
from scripts.refresh_staging_body_initializer import refresh

IMAGE = '136889124221.dkr.ecr.ap-northeast-2.amazonaws.com/standin/inference:' + 'a' * 40


def fixture():
    return {'family': 'StandinStagingAppInferenceTask76CF0AC4', 'taskRoleArn': 'unchanged-role',
            'containerDefinitions': [
                {'name': 'inference', 'image': 'old-live-image', 'environment': [{'name': 'MODE', 'value': 'keep'}]},
                {'name': 'body-assets-init', 'image': 'deleted-image',
                 'command': ['python', '-c', "print('body_assets_ready')"],
                 'secrets': [{'name': 'secret', 'valueFrom': 'unchanged-reference'}]}]}


def test_refresh_changes_only_initializer_image():
    task = fixture(); expected = copy.deepcopy(task)
    expected['containerDefinitions'][1]['image'] = IMAGE
    assert refresh(task, IMAGE) == expected


@pytest.mark.parametrize('image', [IMAGE.rsplit(':', 1)[0] + ':latest', IMAGE.replace('standin/inference', 'standin/bff')])
def test_moving_or_other_images_fail_closed(image):
    with pytest.raises(ValueError): refresh(fixture(), image)


def test_production_and_unknown_initializers_fail_closed():
    task = fixture(); task['family'] = 'StandinAppInferenceTask'
    with pytest.raises(ValueError): refresh(task, IMAGE)
    task = fixture(); task['containerDefinitions'][1]['command'] = ['unknown-command']
    with pytest.raises(ValueError): refresh(task, IMAGE)


def test_deployment_without_optional_initializer_is_unchanged():
    task = fixture(); task['containerDefinitions'].pop()
    expected = copy.deepcopy(task)
    assert refresh(task, IMAGE) == expected


def test_refresh_runs_before_rendering_and_only_on_develop():
    workflow = (Path(__file__).resolve().parents[1]/'.github/workflows/deploy.yml').read_text()
    marker = '- name: Refresh staging body initializer image'
    step = workflow.split(marker, 1)[1].split('\n      - ', 1)[0]
    assert "if: github.ref == 'refs/heads/develop'" in step
    assert 'scripts/refresh_staging_body_initializer.py taskdef.json' in step
    assert workflow.index(marker) < workflow.index('- uses: aws-actions/amazon-ecs-render-task-definition@v1')
