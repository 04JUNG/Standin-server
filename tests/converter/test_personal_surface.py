"""Geometry-only regression cases with no production asset or bone identifiers."""
import unittest

import numpy as np

from converter.local.surface import correct


class PersonalSurfaceTests(unittest.TestCase):
    def setUp(self):
        width = 18
        self.rest = np.array([(i, j, 0.) for i in range(width) for j in range(width)])
        self.faces = []
        for i in range(width - 1):
            for j in range(width - 1):
                a = i * width + j
                self.faces.extend(((a, a+1, a+width), (a+1, a+width+1, a+width)))
        self.faces = np.array(self.faces)
        self.posed = self.rest.copy()
        x, y = (self.rest[:, :2] - 9).T
        self.posed[:, 2] = np.exp(-(x*x+y*y)/32) * np.cos(x) * np.cos(y)

    def test_unnamed_patch_scale_and_coordinate_frame(self):
        fixed, report = correct(self.rest, self.posed, self.faces)
        self.assertTrue(report['applied'])
        self.assertLess(fixed[171, 2], self.posed[171, 2])
        boundary = (self.rest[:, :2] == 0).any(1) | (self.rest[:, :2] == 17).any(1)
        np.testing.assert_array_equal(fixed[boundary], self.posed[boundary])
        angle = .65
        rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
        transform = lambda x: 23*(x@rot) + [100, -10, 8]
        result, _ = correct(transform(self.rest), transform(self.posed), self.faces)
        np.testing.assert_allclose(result, transform(fixed), atol=1e-8)

    def test_rigid_detail_and_disconnected_scales(self):
        detail = self.posed.copy()
        rotation = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        rigid = detail@rotation + [3, 4, 5]
        fixed, report = correct(detail, rigid, self.faces)
        self.assertFalse(report['applied'])
        np.testing.assert_array_equal(fixed, rigid)

        small, _ = correct(self.rest, self.posed, self.faces)
        large = self.rest*90 + 10000
        vertices = np.concatenate((self.rest, large))
        faces = np.concatenate((self.faces, self.faces + len(small)))
        fixed, _ = correct(vertices, np.concatenate((self.posed, large)), faces)
        np.testing.assert_allclose(fixed[:len(small)], small)
        np.testing.assert_array_equal(fixed[len(small):], large)


class SurfaceShadingTests(unittest.TestCase):
    def test_authored_corner_and_geometric_corner(self):
        try:
            import bpy
        except ImportError:
            self.skipTest("Requires Blender")
        from converter.local.surface import normal_policy, update_shading, geometric_normals

        mesh = bpy.data.meshes.new('surface-test')
        try:
            mesh.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)], [],
                             [(0,1,2),(0,2,3)])
            for face in mesh.polygons:
                face.use_smooth = True
            mesh.normals_split_custom_set_from_vertices([(1,0,0),(0,0,1),(0,0,1),(0,0,1)])
            mesh.update()
            policy = normal_policy(mesh)
            ids = np.array([loop.vertex_index for loop in mesh.loops])
            self.assertTrue((policy[ids == 0] == 0).all())
            self.assertTrue((policy[ids != 0] > .99).all())
            mesh.vertices[2].co.z = .6
            mesh.update()
            old = np.array([n.vector[:] for n in mesh.corner_normals])
            update_shading(mesh, policy, np.array([False,False,True,False]))
            new = np.array([n.vector[:] for n in mesh.corner_normals])
            expected = geometric_normals(np.array([v.co[:] for v in mesh.vertices]),
                                         np.array([[0,1,2],[0,2,3]]))
            np.testing.assert_allclose(new[ids == 0], old[ids == 0], atol=.001)
            np.testing.assert_allclose(new[ids != 0], expected[ids[ids != 0]], atol=.001)
        finally:
            bpy.data.meshes.remove(mesh)
