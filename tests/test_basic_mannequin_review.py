from scripts.build_basic_mannequin_review import assemble,strip_images


def fixture():
    return {'study_id':'source','queries':[{'id':'rough_288','input':'raw','overlay':'over','before':[],
         'after':[{'pose_id':'a','camera':{'yaw':3},'matched':'old','canonical':'old2','canonical_kind':'old-kind','features':{'elbows_3d_degrees':[90,120]}}]}]}


def test_only_render_payload_is_removed():
    source=fixture();data=strip_images(source)
    assert data['queries'][0]['after'][0]=={'pose_id':'a','camera':{'yaw':3},'features':{'elbows_3d_degrees':[90,120]}}
    assert source['queries'][0]['after'][0]['matched']=='old'


def test_assets_are_deduplicated_without_changing_candidate_order():
    source=fixture();data=assemble(source,{'a':{'front':'jpeg'},'unused':{'front':'other'}})
    assert list(data['assets'])==['a'] and data['queries'][0]['after'][0]['pose_id']=='a'


def test_focused_review_preserves_existing_input_and_failure():
    source=fixture();data=assemble(source,{'a':{'front':'jpeg'}},('rough_288',))
    assert data['queries'][0]['input']=='raw' and data['queries'][0]['before']==[]
    assert data['source_study_id']=='source' and data['study_id']!='source'
