import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image

from src.description_prefetch import start_description_prefetch
from src.vlm.client import OpenAIVLMClient
from src.vlm.luna_search_description import VERSION, profile


class LunaSearchDescriptionTest(TestCase):
    def test_two_people_share_one_low_call(self):
        settings, _ = profile()
        configuration = settings['catalog']['body_configuration']['both_hands_on_hips']
        row = {
            'description': '양손을 허리에 대고 서 있다.',
            'pose_action': {'postures': [settings['catalog']['posture']['standing']],
                            'actions': [], 'configurations': [configuration]},
            'scene': {'domains': [], 'activities': []},
            'objects': [],
            'meaning': {'intents': [], 'emotions': []},
            'support': {'visible': True, 'visible_scope': 'full', 'hand_observations': [],
                        'certainty': 'medium', 'head_view': 'front',
                        'arm_states': {'left': 'unknown', 'right': 'unknown'}, 'context_evidence': ''},
        }
        generate = Mock(return_value=SimpleNamespace(text=json.dumps({'people': [row, row]})))
        transport = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        image = Image.new('RGB', (100, 100))
        with patch.dict('os.environ', {'ROUTING_LUNA_SIX_FIELD_ENABLED': '1', 'PARALLEL_POSE_DESCRIPTION': '1'}):
            client = OpenAIVLMClient('gpt-6-luna', client=transport)
            result = client.describe_people(image, [(0., 0., .5, 1.), (.5, 0., 1., 1.)])
            prefetch = start_description_prefetch(client, image, None, 100, 100, fullbody=True)
        self.assertEqual(prefetch.status, 'skipped_single_call_contract')
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.descriptions[1].body_details, configuration)
        self.assertEqual(client.description_prompt_version, VERSION)
        generate.assert_called_once()
        call = generate.call_args.kwargs
        self.assertEqual(call['model'], 'gpt-6-luna')
        self.assertEqual(call['config'].reasoning_effort, 'low')
        self.assertEqual(len(call['config'].response_json_schema['properties']['people']['items']['required']), 6)

    def test_invalid_response_uses_one_call(self):
        generate = Mock(return_value=SimpleNamespace(text='{}'))
        transport = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        with patch.dict('os.environ', {'ROUTING_LUNA_SIX_FIELD_ENABLED': '1'}):
            client = OpenAIVLMClient('gpt-6-luna', client=transport)
            result = client.describe_people(Image.new('RGB', (64, 64)), [(0., 0., 1., 1.)])
        self.assertEqual(result.status, 'error')
        generate.assert_called_once()
