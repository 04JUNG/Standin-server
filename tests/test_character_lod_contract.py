"""Fail-closed provenance/correspondence and honest paired-timing boundaries."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
from scripts.character_lod.contract import require_hash,require_subset,require_paired_runs

class CharacterLODContract(unittest.TestCase):
    def test_unapproved_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.blend';p.write_bytes(b'approved')
            digest=hashlib.sha256(p.read_bytes()).hexdigest();require_hash(p,digest)
            p.write_bytes(b'substituted')
            with self.assertRaises(ValueError):require_hash(p,digest)

    def test_protected_connectivity_and_identity(self):
        require_subset([0,1,2,4],[0],[[0,1,2],[1,3,4]],[[0,1,2],[1,2,3]])
        for ids,faces in [([0,1,1],[[0,1,2]]),([1,2,3],[[0,1,2]]),([0,1,2],[[0,1,1]]),([0,1,2],[[0,2,1]])]:
            with self.assertRaises(ValueError):require_subset(ids,[0],[[0,1,2]],faces)

    def test_incomplete_and_mixed_measurements_cannot_support_claim(self):
        rows=[dict(gender='male',case='base',skin='default',variant=v,trial=i,ok=True,controls={'muscle':.5},bvh_sha256='a'*64,report={'shape_bake':{'neutral_preparation':{'status':'prepared'}}}) for v in ('original','candidate') for i in range(3)]
        require_paired_runs(rows)
        with self.assertRaises(ValueError):require_paired_runs(rows[:-1])
        bad=copy.deepcopy(rows);bad[-1]['controls']['muscle']=.8
        with self.assertRaises(ValueError):require_paired_runs(bad)
        bad=copy.deepcopy(rows);bad[-1]['report']['shape_bake']['neutral_preparation']['status']='generated'
        with self.assertRaises(ValueError):require_paired_runs(bad)
        bad=copy.deepcopy(rows);bad[-1]['bvh_sha256']='b'*64
        with self.assertRaises(ValueError):require_paired_runs(bad)

if __name__=='__main__':unittest.main()
