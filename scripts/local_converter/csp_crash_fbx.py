"""Private diagnostic controls; never applied by the product converter.

Preserves every binary property outside cluster Indexes/Weights. Recomputed
node offsets are the only other edits. A no-op rewrite must be byte identical.
Run with a fixed input SHA and a NEW output directory. Raw assets stay local.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

MAGIC = b'Kaydara FBX Binary  \x00\x1a\x00'
MAX_BYTES = 128 * 1024**2
MAX_EXPANDED = 64 * 1024**2


@dataclass
class Node:
    name: bytes
    props: list[bytes] = field(default_factory=list)
    children: list['Node'] = field(default_factory=list)
    sentinel: bytes = b''

    def child(self, name):
        return next((n for n in self.children if n.name == name), None)


def scalar(raw):
    kind = raw[:1]
    if kind == b'S':
        return raw[5:]
    return struct.unpack({b'L':'<q', b'I':'<i', b'D':'<d'}[kind], raw[1:])[0]


def array(raw):
    kind = raw[:1]
    count, encoding, size = struct.unpack_from('<III', raw, 1)
    stride, fmt = {b'i':(4,'i'), b'd':(8,'d')}[kind]
    expected = count * stride
    if expected > MAX_EXPANDED or size != len(raw)-13 or encoding not in (0, 1):
        raise ValueError('Invalid array size/encoding')
    data = raw[13:]
    if encoding:
        decoder = zlib.decompressobj()
        data = decoder.decompress(data, expected + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Invalid compressed array')
    if len(data) != expected:
        raise ValueError('Array length mismatch')
    return list(struct.unpack('<'+str(count)+fmt, data))


def pack_array(kind, values):
    fmt = {b'i':'i', b'd':'d'}[kind]
    payload = struct.pack('<'+str(len(values))+fmt, *values)
    compressed = zlib.compress(payload)
    return kind + struct.pack('<III', len(values), 1, len(compressed)) + compressed


class Document:
    def __init__(self, data):
        if len(data) > MAX_BYTES or data[:23] != MAGIC:
            raise ValueError('Expected bounded binary FBX')
        self.data = data
        self.version = struct.unpack_from('<I', data, 23)[0]
        if self.version not in (7400, 7500, 7700):
            raise ValueError('Unsupported FBX version')
        self.header = '<QQQB' if self.version >= 7500 else '<IIIB'
        self.width = struct.calcsize(self.header)
        self.nodes = []
        at = 27
        while any(data[at:at+self.width]):
            n, at = self.read(at, 0)
            self.nodes.append(n)
        self.tail = data[at:]

    def read(self, at, depth):
        if depth > 64 or at+self.width > len(self.data):
            raise ValueError('Invalid node depth/header')
        end, count, size, length = struct.unpack_from(self.header, self.data, at)
        start = at+self.width
        at = start+length
        if not at <= at+size <= end <= len(self.data):
            raise ValueError('Invalid node offsets')
        n = Node(self.data[start:at])
        prop_end = at+size
        if count > size:
            raise ValueError('Invalid property count')
        for _ in range(count):
            begin = at
            kind = self.data[at:at+1]; at += 1
            fixed = {b'Y':2,b'C':1,b'I':4,b'F':4,b'D':8,b'L':8,b'B':1}
            if kind in fixed:
                at += fixed[kind]
            elif kind in (b'S',b'R'):
                at += 4 + struct.unpack_from('<I', self.data, at)[0]
            elif kind in (b'f',b'd',b'l',b'i',b'b',b'c'):
                at += 12 + struct.unpack_from('<I', self.data, at+8)[0]
            else:
                raise ValueError('Unsupported property type')
            if at > prop_end:
                raise ValueError('Property outside block')
            n.props.append(self.data[begin:at])
        if at != prop_end:
            raise ValueError('Property size mismatch')
        while at < end:
            if end-at == self.width and not any(self.data[at:end]):
                n.sentinel = self.data[at:end]; at=end; break
            ch, at = self.read(at, depth+1); n.children.append(ch)
        if at != end:
            raise ValueError('Node end mismatch')
        return n, end

    def write(self):
        def emit(n, at):
            props = b''.join(n.props)
            prefix = self.width+len(n.name)+len(props)
            children = bytearray()
            for ch in n.children:
                children.extend(emit(ch, at+prefix+len(children)))
            body = n.name+props+children+n.sentinel
            return struct.pack(self.header, at+self.width+len(body), len(n.props), len(props),len(n.name))+body
        out=bytearray(self.data[:27])
        for n in self.nodes:
            out.extend(emit(n,len(out)))
        # Preserve opaque footer IDs and maintain the FBX 16-byte alignment.
        # Version + 120 zeros + fixed magic form the final 140-byte suffix.
        prefix = self.tail[:self.width+20]
        padding = 16 - ((len(out)+len(prefix)) % 16)
        tail = prefix + b'\0'*padding + self.tail[-140:]
        return bytes(out)+tail

    def root(self, name):
        return next(n for n in self.nodes if n.name == name)

    def properties(self):
        result={}
        def walk(nodes,path):
            for i,n in enumerate(nodes):
                key=f'{path}/{i}:{n.name.decode("ascii",errors="replace")}'
                result[key]=n.props
                walk(n.children,key)
        walk(self.nodes,'')
        return result

    def clusters(self):
        return [n for n in self.root(b'Objects').children if n.name==b'Deformer' and scalar(n.props[2])==b'Cluster']


def influences(doc):
    # Group per geometry through Skin connections; do not conflate vertex IDs
    # from different meshes. Reject ambiguous/missing relationships.
    objects={scalar(n.props[0]):n for n in doc.root(b'Objects').children}
    parents={}
    for n in doc.root(b'Connections').children:
        if n.name==b'C' and scalar(n.props[0])==b'OO':
            parents.setdefault(scalar(n.props[1]),[]).append(scalar(n.props[2]))
    rows={}; clusters=doc.clusters()
    for ci,n in enumerate(clusters):
        skins=[i for i in parents.get(scalar(n.props[0]),[]) if i in objects and objects[i].name==b'Deformer' and scalar(objects[i].props[2])==b'Skin']
        if len(skins)!=1: raise ValueError('Ambiguous cluster skin')
        geos=[i for i in parents.get(skins[0],[]) if i in objects and objects[i].name==b'Geometry']
        if len(geos)!=1: raise ValueError('Ambiguous skin geometry')
        ix,wt=n.child(b'Indexes'),n.child(b'Weights')
        if ix is None and wt is None: continue
        if ix is None or wt is None: raise ValueError('Incomplete cluster')
        indices,weights=array(ix.props[0]),array(wt.props[0])
        if len(indices)!=len(weights): raise ValueError('Cluster pair mismatch')
        seen=set()
        for vi,w in zip(indices,weights):
            if vi<0 or vi in seen or not math.isfinite(w) or w<0: raise ValueError('Invalid weight')
            seen.add(vi)
            if w>0: rows.setdefault((geos[0],vi),[]).append((ci,w))
    return clusters,rows


def limit_influences(doc, maximum):
    if not 1<=maximum<=9: raise ValueError('Invalid influence limit')
    clusters,rows=influences(doc)
    changes={}; removed=[]
    for key,row in rows.items():
        if len(row)<=maximum: continue
        ordered=sorted(row,key=lambda x:(-x[1],x[0])); keep=ordered[:maximum]
        total=sum(w for _,w in keep)
        for ci,w in row:
            changes[(ci,key[1])]=w/total if (ci,w) in keep else None
        removed.append(sum(w for _,w in ordered[maximum:]))
    for ci,n in enumerate(clusters):
        ix,wt=n.child(b'Indexes'),n.child(b'Weights')
        if ix is None: continue
        indices,weights=array(ix.props[0]),array(wt.props[0]); newi=[]; neww=[]
        if not any((ci,vi) in changes for vi in indices): continue
        for vi,w in zip(indices,weights):
            value=changes.get((ci,vi),w)
            if value is not None: newi.append(vi); neww.append(value)
        ix.props=[pack_array(b'i',newi)]; wt.props=[pack_array(b'd',neww)]
    return {'changedVertices':len(removed),'maxRemovedWeightSum':max(removed,default=0),
            'meanRemovedWeightSum':sum(removed)/len(removed) if removed else 0}


def sha(data): return hashlib.sha256(data).hexdigest()


def make(source, expected_sha, output):
    data=source.read_bytes()
    if sha(data)!=expected_sha: raise ValueError('Source SHA mismatch')
    output.mkdir(parents=True,exist_ok=False)
    original=Document(data); props=original.properties()
    assert original.write()==data, 'No-op writer is not byte identical'
    _,rows=influences(original)
    report={'schemaVersion':1,'diagnosticOnly':True,'sourceSha256':sha(data),
            'sourceMaxInfluences':max(map(len,rows.values())),
            'sourceVerticesOverEight':sum(len(r)>8 for r in rows.values()),'variants':[]}
    for name,limit in [('original9',None),('binary-roundtrip9',None),('max8',8),('max7',7),('restored9',None)]:
        doc=Document(data); metrics=limit_influences(doc,limit) if limit else {}
        result=doc.write(); after=Document(result)
        diff=[p for p,v in after.properties().items() if v!=props[p]]
        assert all(p.endswith((':Indexes',':Weights')) for p in diff)
        assert set(props)==set(after.properties()), 'Structure changed'
        _,newrows=influences(after)
        if limit: assert max(map(len,newrows.values()))<=limit
        else: assert result==data
        unchanged=[k for k,row in rows.items() if limit is None or len(row)<=limit]
        assert all(rows[k]==newrows[k] for k in unchanged)
        dest=output/(name+'.fbx'); dest.write_bytes(result)
        report['variants'].append({'file':dest.name,'sha256':sha(result),'bytes':len(result),
            'maxInfluences':max(map(len,newrows.values())),'changedProperties':diff,
            'allOtherPropertyBytesExact':True,'unaffectedVerticesExact':True,**metrics})
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    r=make(a.source,a.sha256,a.output)
    print(json.dumps({**r,'variants':[{k:v for k,v in x.items() if k!='changedProperties'} for x in r['variants']]},indent=2))
