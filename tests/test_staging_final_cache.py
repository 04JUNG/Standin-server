import copy
import importlib.util
import json
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cache_deploy', root/'scripts/prepare_staging_final_cache.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
config = json.loads((root/'config/staging-final-cache.json').read_text())


def task():
 return {'family':'StandinStagingAppConverterTask', 'cpu':'2048', 'containerDefinitions':[
  {'name':'converter','image':'unchanged','environment':[{'name':'CONVERTER_SHAPE_REGISTRY','value':'unchanged-registry'}],
   'secrets':[{'name':'PRIVATE_TOKEN','valueFrom':'unchanged-reference'}]},
  {'name':'body-assets-init','image':'unchanged-init'}]}


def test_staging_cache_and_kill_switch_preserve_all_runtime_wiring():
 source = task();before = copy.deepcopy(source);result = module.prepare(source,config)
 assert source == before
 expected = copy.deepcopy(source)
 expected['containerDefinitions'][0]['environment'] = result['containerDefinitions'][0]['environment']
 assert result == expected
 env = {r['name']:r['value'] for r in result['containerDefinitions'][0]['environment']}
 assert env['CONVERTER_FINAL_LIBRARY_SHA256'] == config['librarySha256']
 assert env['CONVERTER_FINAL_PREWARM_ENABLED'] == '0'
 assert env['CONVERTER_REUSE_REST_METRICS'] == '1'
 disabled = module.prepare(result,config,'0','0')
 env = {r['name']:r['value'] for r in disabled['containerDefinitions'][0]['environment']}
 assert env['CONVERTER_FINAL_CACHE_ROOT'] == '' and env['CONVERTER_REUSE_REST_METRICS'] == '0'
 assert module.prepare(result,config) == result


@pytest.mark.parametrize('problem',['production','wrong_bucket','unsafe_control','duplicate_env','secret','missing','invalid_flag'])
def test_unsafe_configuration_is_rejected(problem):
 source = task();c = copy.deepcopy(config);enabled = '1'
 if problem == 'production': source['family'] = 'StandinAppConverterTask'
 if problem == 'wrong_bucket': c['bucket'] = 'production-bucket'
 if problem == 'unsafe_control': c['libraryKey'] = c['dataPrefix'] + 'control.json'
 if problem == 'duplicate_env': source['containerDefinitions'][0]['environment'] *= 2
 if problem == 'secret': source['containerDefinitions'][0]['secrets'].append({'name':'CONVERTER_FINAL_CACHE_ROOT'})
 if problem == 'missing': source['containerDefinitions'] = []
 if problem == 'invalid_flag': enabled = 'true'
 with pytest.raises(ValueError): module.prepare(source,c,enabled)
