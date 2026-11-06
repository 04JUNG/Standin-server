"""Native OpenGL implementation of the approved comparison renderer.

Only surface appearance changes. Geometry comes from reviewed FBX/GLB assets;
the eight native azimuths retain the directional-poster camera convention.
"""
from __future__ import annotations

import itertools
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

VERTEX = '''#version 330
uniform mat4 mvp;
uniform mat3 normal_matrix;
in vec3 position;
in vec3 normal;
out vec3 n;
void main(){n=normalize(normal_matrix*normal);gl_Position=mvp*vec4(position,1.);}
'''
COLOR = '''#version 330
in vec3 n;out vec4 color;
void main(){vec3 N=normalize(n);
float key=clamp((dot(N,vec3(-.4,.5,.768115))+.75)/1.75,0.,1.);
float fill=clamp((dot(N,vec3(.8,0.,.6))+.85)/1.85,0.,1.);
color=vec4(.72*(vec3(.28,.29,.31)+vec3(.38,.39,.41)*key+vec3(.13)*fill),1.);}
'''
NORMAL = '''#version 330
in vec3 n;out vec4 color;
void main(){color=vec4(normalize(n)*.5+.5,gl_FragCoord.z);}
'''
POST_VERTEX = '''#version 330
in vec2 position;out vec2 uv;
void main(){uv=position*.5+.5;gl_Position=vec4(position,0.,1.);}
'''
POST = '''#version 330
uniform sampler2D color_map, normal_map;
uniform float size;
out vec4 color;
vec2 pixelUV(){return (vec2(ivec2(gl_FragCoord.xy)%ivec2(int(size)))+.5)/size;}
float edgeAt(float radius){
 if(radius<=0.)return 0.;
 vec2 uv=pixelUV();
 vec4 center=texture(normal_map,uv);float edge=0.;bool inside=center.a>.01;
 for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++){
  if(x==0&&y==0)continue;
  vec4 other=texture(normal_map,uv+vec2(x,y)*radius/size);
  bool neighbor=other.a>.01;
  if(inside!=neighbor)edge=1.;
  else if(inside){
   float nd=1.-dot(normalize(center.rgb*2.-1.),normalize(other.rgb*2.-1.));
   float dd=abs(center.a-other.a);
   edge=max(edge,max(smoothstep(.12,.36,nd),smoothstep(.003,.010,dd)));
  }
 }
 return edge;
}
void main(){
 vec2 uv=pixelUV();
 float width=.65*size/512.;
 float edge=mix(edgeAt(floor(width)),edgeAt(ceil(width)),fract(width));
 vec3 base=mix(texture(color_map,uv).rgb,vec3(1.),.22);
 if(texture(normal_map,uv).a>.01)base=clamp((base-.62)*2.6+.62,.06,.95);
 vec3 c=mix(base,vec3(.018,.023,.025),edge*.94);
 c=mix(1.055*pow(c,vec3(1./2.4))-.055,c*12.92,lessThanEqual(c,vec3(.0031308)));
 color=vec4(c,1.);
}
'''


def load_glb(path: Path):
    """Read the bounded static GLB format emitted by converter.preview_model."""
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', raw)
    if (magic, version, length) != (0x46546C67, 2, len(raw)):
        raise ValueError(f'invalid GLB: {path}')
    count, kind = struct.unpack_from('<II', raw, 12)
    if kind != 0x4E4F534A:
        raise ValueError('missing GLB JSON')
    doc = json.loads(raw[20:20 + count])
    size, kind = struct.unpack_from('<II', raw, 20 + count)
    binary = raw[28 + count:28 + count + size]
    if kind != 0x004E4942 or doc.get('skins') or doc.get('animations'):
        raise ValueError('expected baked static GLB')
    if doc['nodes'] != [{'mesh': 0}] or len(doc['meshes']) != 1:
        raise ValueError('unexpected scene transforms')

    def accessor(index):
        a = doc['accessors'][index]
        view = doc['bufferViews'][a['bufferView']]
        dtype = {5126: '<f4', 5125: '<u4', 5123: '<u2'}[a['componentType']]
        width = {'VEC3': 3, 'SCALAR': 1}[a['type']]
        if view.get('byteStride') or a.get('sparse') or view['buffer'] != 0:
            raise ValueError('unsupported accessor layout')
        return np.frombuffer(binary, dtype=dtype, count=a['count'] * width,
                             offset=view.get('byteOffset', 0) + a.get('byteOffset', 0)).reshape(-1, width)

    result = []
    for p in doc['meshes'][0]['primitives']:
        if p.get('mode', 4) != 4:
            raise ValueError('expected triangles')
        pos, normal = (accessor(p['attributes'][key]) for key in ('POSITION', 'NORMAL'))
        indices = accessor(p['indices']).reshape(-1).astype('u4')
        if not np.isfinite(pos).all() or not np.isfinite(normal).all() or indices.max() >= len(pos):
            raise ValueError('invalid surface')
        result.append((pos, normal, indices))
    return result


class Renderer:
    def __init__(self, size=512):
        import moderngl as gl
        self.gl, self.size = gl, size
        self.ctx = c = gl.create_standalone_context(require=330)
        self.programs = [c.program(vertex_shader=VERTEX, fragment_shader=f) for f in (COLOR, NORMAL)]
        self.color = c.texture((size, size), 4)
        # Alpha carries gl_FragCoord.z for the outline pass. An 8-bit texture
        # quantizes smooth depth into 1/255 steps, exceeding edgeAt's .003
        # threshold and drawing false contours across otherwise smooth faces.
        self.normal = c.texture((size, size), 4, dtype='f4')
        self.color.repeat_x = self.color.repeat_y = False
        self.normal.repeat_x = self.normal.repeat_y = False
        self.normal.filter = (gl.NEAREST, gl.NEAREST)
        self.resolved = c.framebuffer([self.color])
        self.msaa = c.framebuffer([c.renderbuffer((size, size), 4, samples=4)],
                                 c.depth_renderbuffer((size, size), samples=4))
        self.normal_fb = c.framebuffer([self.normal], c.depth_renderbuffer((size, size)))
        self.output = c.simple_framebuffer((size, size), components=3)
        self.atlas = c.simple_framebuffer((size * 8, size), components=3)
        self.post = c.program(vertex_shader=POST_VERTEX, fragment_shader=POST)
        self.post['size'] = float(size)
        self.post['color_map'], self.post['normal_map'] = 0, 1
        self.quad_buffer = c.buffer(np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4').tobytes())
        self.quad = c.vertex_array(self.post, [(self.quad_buffer, '2f', 'position')])
        self.resources = []

    def load(self, parts):
        for resource in self.resources:
            resource.release()
        self.resources, self.vaos = [], [[], []]
        self.low = np.min([p[0].min(axis=0) for p in parts], axis=0)
        self.high = np.max([p[0].max(axis=0) for p in parts], axis=0)
        for pos, normal, indices in parts:
            vbo = self.ctx.buffer(np.column_stack((pos, normal)).astype('f4').tobytes())
            ibo = self.ctx.buffer(indices.astype('u4').tobytes())
            self.resources.extend([vbo, ibo])
            for i, program in enumerate(self.programs):
                vao = self.ctx.vertex_array(program, [(vbo, '3f 3f', 'position', 'normal')], ibo)
                self.resources.append(vao)
                self.vaos[i].append(vao)

    def render(self, angle, *, comparison_pitch=False, atlas_index=None):
        c, gl = self.ctx, self.gl
        a = math.radians(angle)
        view = np.array([math.sin(a), .02667 if comparison_pitch else 0, math.cos(a)])
        view /= np.linalg.norm(view)
        right = np.cross([0, 1, 0], view); right /= np.linalg.norm(right)
        up = np.cross(view, right)
        axes = np.array([right, up, view])
        center = (self.low + self.high) * .5
        corners = np.array(list(itertools.product(*zip(self.low, self.high)))) - center
        extent = max(np.abs(corners @ axes[:2].T).max(), .05) * 2 * 1.13
        mv = np.eye(4)
        mv[:3, :3] = axes
        mv[:3, 3] = -axes @ (center + view * extent * 3)
        near, far = extent * .1, extent * 6
        projection = np.diag([2/extent, 2/extent, -2/(far-near), 1.])
        projection[2, 3] = -(far+near)/(far-near)
        for p in self.programs:
            p['mvp'].write((projection @ mv).astype('f4').T.tobytes())
            p['normal_matrix'].write(axes.astype('f4').T.tobytes())
        c.enable(gl.DEPTH_TEST | gl.CULL_FACE)
        for i, fb in enumerate((self.msaa, self.normal_fb)):
            fb.use()
            fb.clear(*( (1,1,1,1) if i == 0 else (0,0,0,0)))
            for vao in self.vaos[i]:
                vao.render()
        c.copy_framebuffer(self.resolved, self.msaa)
        c.disable(gl.DEPTH_TEST | gl.CULL_FACE)
        target = self.output if atlas_index is None else self.atlas
        target.use()
        if atlas_index is not None:
            c.viewport = (atlas_index * self.size, 0, self.size, self.size)
        self.color.use(0); self.normal.use(1)
        self.quad.render(gl.TRIANGLE_STRIP)
        if atlas_index is not None:
            return None
        return Image.frombytes('RGB', (self.size, self.size), self.output.read(components=3, alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def render_views(self, angles):
        """One GPU readback for eight cameras; no per-camera GPU/CPU stall."""
        if len(angles) != 8:
            raise ValueError('the atlas requires exactly eight views')
        for index, angle in enumerate(angles):
            self.render(angle, atlas_index=index)
        pixels = self.atlas.read(components=3, alignment=1)
        image = Image.frombytes('RGB', (self.size * 8, self.size), pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return [image.crop((index*self.size, 0, (index+1)*self.size, self.size)) for index in range(8)]
