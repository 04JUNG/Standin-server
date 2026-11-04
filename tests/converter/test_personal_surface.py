"""Geometry-only regression cases with no production asset or bone identifiers."""
import unittest

import numpy as np

from converter.local.surface import Neighborhood, correct


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

    def test_one_degenerate_reference_face_does_not_disable_the_body(self):
        faces = np.vstack((self.faces, [0, 0, 1]))
        fixed, report = correct(self.rest, self.posed, faces)
        self.assertTrue(report["applied"])
        np.testing.assert_array_equal(fixed[[0, 1]], self.posed[[0, 1]])
        self.assertLess(fixed[171, 2], self.posed[171, 2])

    def test_hundreds_of_static_islands_do_not_hide_deformed_skin(self):
        props = np.array([[[i+30, 0, 0], [i+30.4, 0, 0], [i+30, .4, 0]]
                          for i in range(300)]).reshape(-1, 3)
        faces = np.vstack((self.faces, np.arange(len(props)).reshape(-1, 3)+len(self.rest)))
        fixed, report = correct(np.vstack((self.rest, props)), np.vstack((self.posed, props)), faces)
        self.assertTrue(report["applied"])
        np.testing.assert_array_equal(fixed[len(self.rest):], props)
        np.testing.assert_allclose(fixed[:len(self.rest)], correct(self.rest, self.posed, self.faces)[0])

    def test_diffusion_radius_survives_grid_refinement(self):
        # Sample the same physical ripple on two independently triangulated meshes.
        amplitudes = []
        for n in (33, 65):
            axis = np.linspace(-1., 1., n)
            x, y = np.meshgrid(axis, axis)
            rest = np.column_stack((x.ravel(), y.ravel(), np.zeros(n*n)))
            field = (np.exp(-12*(x*x+y*y))*np.cos(16*x)*np.cos(16*y)).reshape(-1, 1)
            ids = np.arange(n*n).reshape(n, n)[:-1, :-1].ravel()
            faces = np.vstack((np.column_stack((ids, ids+1, ids+n+1)),
                               np.column_stack((ids, ids+n+1, ids+n))))
            amplitudes.append(Neighborhood(faces, rest).diffuse(field, .15)[n*n//2, 0])
        self.assertLess(abs(amplitudes[0] / amplitudes[1] - 1), .08)
        self.assertTrue(all(.1 < value < .25 for value in amplitudes))


class SurfaceShadingTests(unittest.TestCase):
    def test_registered_quad_and_authored_sharp_edge_anchors(self):
        try:
            import bpy
        except ImportError:
            self.skipTest("Requires Blender")
        from converter.local.surface import prepare_reference

        mesh = bpy.data.meshes.new("arbitrary-quad-grid")
        try:
            points = [(i,j,0.) for i in range(4) for j in range(4)]
            faces = [(k,k+1,k+5,k+4) for k in range(12) if k%4 < 3]
            mesh.from_pydata(points, [], faces)
            sharp = mesh.attributes.new("sharp_edge", "BOOLEAN", "EDGE")
            marked = next(e for e in mesh.edges if set(e.vertices) == {5,6})
            sharp.data[marked.index].value = True
            mesh.update()
            reference = prepare_reference(mesh)
            self.assertEqual(len(reference.triangles), 18)
            self.assertTrue(reference.anchors[[0,5,6]].all())
            self.assertFalse(reference.anchors[9])
        finally:
            bpy.data.meshes.remove(mesh)

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
