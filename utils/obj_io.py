from __future__ import print_function, absolute_import, division
import numpy as np


def load_obj_data(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        if len(line) < 2:
            continue
        line_data = line.strip().split(' ')
        if '' in line_data:
            line_data.remove('')
        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))
            else:
                vc_list.append((0.5, 0.5, 0.5))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            if len(line_data) < 3:
                continue
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])

            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model


def load_obj_data_binary(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'rb')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line_data = line.strip().split(' ')

        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))
            else:
                vc_list.append((0.5, 0.5, 0.5))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model


def save_obj_data(model, filename):
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'w') as fp:
        if 'v' in model and model['v'].size != 0:
            for i in range(model['v'].shape[0]):
                v = model['v'][i]
                fp.write('v %f %f %f ' % (v[0], v[1], v[2]))
                if 'c' in model and model['c'].size != 0:
                    c = model['c'][i]
                    fp.write('%f %f %f\n' % (c[0], c[1], c[2]))
                else:
                    fp.write('\n')

        if 'vn' in model and model['vn'].size != 0:
            for vn in model['vn']:
                fp.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))

        if 'vt' in model and model['vt'].size != 0:
            for vt in model['vt']:
                fp.write('vt %f %f\n' % (vt[0], vt[1]))

        if 'f' in model and model['f'].size != 0:
            if 'fn' in model and model['fn'].size != 0 and 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['fn'].size
                assert model['f'].size == model['ft'].size
                for f_, ft_, fn_ in zip(model['f'], model['ft'], model['fn']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                             (f[0], ft[0], fn[0], f[1], ft[1], fn[1], f[2], ft[2], fn[2]))
            elif 'fn' in model and model['fn'].size != 0:
                assert model['f'].size == model['fn'].size
                for f_, fn_ in zip(model['f'], model['fn']):
                    f = np.copy(f_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d//%d %d//%d %d//%d\n' % (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
            elif 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['ft'].size
                for f_, ft_ in zip(model['f'], model['ft']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
            else:
                for f_ in model['f']:
                    f = np.copy(f_) + 1
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def save_obj_data_binary(model, filename):
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'wb') as fp:
        if 'v' in model and model['v'].size != 0:
            for v in model['v']:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if 'vn' in model and model['vn'].size != 0:
            for vn in model['vn']:
                fp.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))

        if 'vt' in model and model['vt'].size != 0:
            for vt in model['vt']:
                fp.write('vt %f %f\n' % (vt[0], vt[1]))

        if 'f' in model and model['f'].size != 0:
            if 'fn' in model and model['fn'].size != 0 and 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['fn'].size
                assert model['f'].size == model['ft'].size
                for f_, ft_, fn_ in zip(model['f'], model['ft'], model['fn']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                             (f[0], ft[0], fn[0], f[1], ft[1], fn[1], f[2], ft[2], fn[2]))
            elif 'fn' in model and model['fn'].size != 0:
                assert model['f'].size == model['fn'].size
                for f_, fn_ in zip(model['f'], model['fn']):
                    f = np.copy(f_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d//%d %d//%d %d//%d\n' % (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
            elif 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['ft'].size
                for f_, ft_ in zip(model['f'], model['ft']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
            else:
                for f_ in model['f']:
                    f = np.copy(f_) + 1
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def save_obj_data_binary_for_voxelization(model, min_corner, max_corner, corner_size, filename):
    """
    Save Obj data with the corner of the bounding box
    This small feature will be useful when you use some 3d software to voxelize the mesh,
    because it can help you avoid complex alignments.
    """
    assert 'v' in model and model['v'].size != 0
    with open(filename, 'wb') as fp:
        if 'v' in model and model['v'].size != 0:
            for v in model['v']:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        # add corner
        fp.write('v %f %f %f\n' % (min_corner[0] + corner_size, min_corner[1], min_corner[2]))
        fp.write('v %f %f %f\n' % (min_corner[0], min_corner[1] + corner_size, min_corner[2]))
        fp.write('v %f %f %f\n' % (min_corner[0], min_corner[1], min_corner[2] + corner_size))
        fp.write('v %f %f %f\n' % (min_corner[0], min_corner[1], min_corner[2]))

        fp.write('v %f %f %f\n' % (max_corner[0] - corner_size, max_corner[1], max_corner[2]))
        fp.write('v %f %f %f\n' % (max_corner[0], max_corner[1] - corner_size, max_corner[2]))
        fp.write('v %f %f %f\n' % (max_corner[0], max_corner[1], max_corner[2] - corner_size))
        fp.write('v %f %f %f\n' % (max_corner[0], max_corner[1], max_corner[2]))

        # if 'vn' in model and model['vn'].size != 0:
        #     for vn in model['vn']:
        #         fp.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))
        #
        # if 'vt' in model and model['vt'].size != 0:
        #     for vt in model['vt']:
        #         fp.write('vt %f %f\n' % (vt[0], vt[1]))

        if 'f' in model and model['f'].size != 0:
            # if 'fn' in model and model['fn'].size != 0 and 'ft' in model and model['ft'].size != 0:
            #     assert model['f'].size == model['fn'].size
            #     assert model['f'].size == model['ft'].size
            #     for f_, ft_, fn_ in zip(model['f'], model['ft'], model['fn']):
            #         f = np.copy(f_) + 1
            #         ft = np.copy(ft_) + 1
            #         fn = np.copy(fn_) + 1
            #         fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
            #                  (f[0], ft[0], fn[0], f[1], ft[1], fn[1], f[2], ft[2], fn[2]))
            # elif 'fn' in model and model['fn'].size != 0:
            #     assert model['f'].size == model['fn'].size
            #     for f_, fn_ in zip(model['f'], model['fn']):
            #         f = np.copy(f_) + 1
            #         fn = np.copy(fn_) + 1
            #         fp.write('f %d//%d %d//%d %d//%d\n' % (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
            # elif 'ft' in model and model['ft'].size != 0:
            #     assert model['f'].size == model['ft'].size
            #     for f_, ft_ in zip(model['f'], model['ft']):
            #         f = np.copy(f_) + 1
            #         ft = np.copy(ft_) + 1
            #         fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
            # else:
            for f_ in model['f']:
                f = np.copy(f_) + 1
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

        # add corner
        vid_start = model['v'].size / 3
        fp.write('f %d %d %d\n' % (vid_start + 1, vid_start + 2, vid_start + 3))
        fp.write('f %d %d %d\n' % (vid_start + 1, vid_start + 4, vid_start + 2))
        fp.write('f %d %d %d\n' % (vid_start + 1, vid_start + 3, vid_start + 4))
        fp.write('f %d %d %d\n' % (vid_start + 2, vid_start + 4, vid_start + 3))

        vid_start += 4
        fp.write('f %d %d %d\n' % (vid_start + 1, vid_start + 2, vid_start + 3))
        fp.write('f %d %d %d\n' % (vid_start + 1, vid_start + 4, vid_start + 2))
        fp.write('f %d %d %d\n' % (vid_start + 1, vid_start + 3, vid_start + 4))
        fp.write('f %d %d %d\n' % (vid_start + 2, vid_start + 4, vid_start + 3))


def save_mesh_as_obj(path, vertices, faces=None, normals=None, colors=None):
    mesh = dict()
    mesh['v'] = vertices
    if faces is not None:
        mesh['f'] = faces
    if normals is not None:
        mesh['vn'] = normals
    if colors is not None:
        mesh['c'] = colors
    save_obj_data(mesh, path)


# def save_mesh_as_ply(path, vertices, faces=None, normals=None, colors=None):
#     path = path + '.ply'
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     if faces is not None:
#         mesh.triangles = o3d.utility.Vector3iVector(faces)
#     if normals is not None:
#         mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
#     if colors is not None:
#         mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
#     o3d.io.write_triangle_mesh(path, mesh, write_vertex_colors=True if colors is not None else False)


def save_mesh_as_ply(path, vertices, faces = None, normals = None, colors = None):
    import struct
    fp = open(path, 'w')
    fp.write('ply\n')
    fp.write('format binary_little_endian 1.0\n')
    fp.write('element vertex %d\n' % vertices.shape[0])
    fp.write('property float x\nproperty float y\nproperty float z\n')
    if normals is not None:
        fp.write('property float nx\nproperty float ny\nproperty float nz\n')
    if colors is not None:
        fp.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
    # if faces is not None:
    face_num = 0 if faces is None else faces.shape[0]
    fp.write('element face %d\n' % face_num)
    fp.write('property list int int vertex_indices\n')
    fp.write('end_header\n')
    fp.close()

    fp = open(path, 'ab')
    # data = vertices.astype(np.float32)
    # if normals is not None:
    #     normals = normals.astype(np.float32)
    #     data = np.concatenate([data, normals], 1)

    # if colors is not None:
    #     colors = (colors*255).astype(np.uint8)
    #     data = np.concatenate([data, colors], 1)
    # data = data.reshape(-1)
    # data_num = data.shape[0]
    # data = data.astype('<u2').tostring()
    # fp.write(data)
    # fp.close()
    vertices = vertices.astype(np.float32)
    if normals is not None:
        normals = normals.astype(np.float32)
    if colors is not None:
        if np.min(colors) >= 0. and np.max(colors) <= 1.:
            colors *= 255
        colors = np.clip(colors, 0, 255)
        colors = colors.astype(np.uint8)
    for i in range(vertices.shape[0]):
        v = vertices[i]
        n = normals[i] if normals is not None else None
        c = colors[i] if colors is not None else None
        if normals is None:
            if colors is None:
                data = struct.pack('3f', v[0], v[1], v[2])
            else:
                data = struct.pack('3f3B', v[0], v[1], v[2], c[0], c[1], c[2])
        else:
            if colors is None:
                data = struct.pack('6f', v[0], v[1], v[2], n[0], n[1], n[2])
            else:
                data = struct.pack('6f3B', v[0], v[1], v[2], n[0], n[1], n[2], c[0], c[1], c[2])
        fp.write(data)
    if faces is not None:
        for i in range(faces.shape[0]):
            f = faces[i]
            data = struct.pack('4i', 3, f[0], f[1], f[2])
            fp.write(data)
    fp.close()


def save_corres(path, pts1, pts2):
    """
    source is red, target is blue
    """
    pts = np.concatenate([pts1, pts2], axis = 1).reshape(-1, 3)
    lines = np.arange(0, pts.shape[0], dtype = np.int).reshape(-1, 2)

    with open(path, 'w') as fp:
        for vi in range(pts.shape[0]):
            p = pts[vi]
            if vi % 2 == 0:
                fp.write('v %f %f %f 1 0 0\n' % (p[0], p[1], p[2]))
            else:
                fp.write('v %f %f %f 0 0 1\n' % (p[0], p[1], p[2]))
        for li in range(lines.shape[0]):
            fp.write('l %d %d\n' % (lines[li, 0] + 1, lines[li, 1] + 1))
