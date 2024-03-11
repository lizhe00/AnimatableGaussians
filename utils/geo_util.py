import torch
import torch.nn.functional as F
import numpy as np

from utils.knn import knn_gather


def barycentric_coordinate(pts, face_vertices):
    """
    :param pts: (B, N, 3)
    :param face_vertices: (B, N, 3, 3)
    :return bc_coords: (B, N, 3)
    """
    vec0 = face_vertices[:, :, 0] - pts
    vec1 = face_vertices[:, :, 1] - pts
    vec2 = face_vertices[:, :, 2] - pts
    area0 = torch.linalg.norm(torch.cross(vec1, vec2), dim = -1)
    area1 = torch.linalg.norm(torch.cross(vec2, vec0), dim = -1)
    area2 = torch.linalg.norm(torch.cross(vec0, vec1), dim = -1)
    bc_coord = torch.stack([area0, area1, area2], -1)
    bc_coord = F.normalize(bc_coord, p = 1, dim = -1, eps = 1e-16)
    return bc_coord


def barycentric_interpolate(vert_attris, faces, face_ids, bc_coords):
    """
    :param vert_attris: (B, V, C)
    :param faces: (B, F, 3)
    :param face_ids: (B, N)
    :param bc_coords: (B, N, 3)
    :return inter_attris: (B, N, C)
    """
    selected_faces = torch.gather(faces, 1, face_ids.unsqueeze(-1).expand(-1, -1, 3))  # (B, N, 3)
    face_attris = knn_gather(vert_attris, selected_faces)  # (B, N, 3, C)
    inter_attris = (face_attris * bc_coords.unsqueeze(-1)).sum(-2)  # (B, N, C)
    return inter_attris


def sample_surface_pts(mesh, count, mask = None, w_color = False):
    """
    Modified from Scanimate code
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """
    valid_faces = mesh.faces[mask]
    face_index = np.random.choice(a = valid_faces.shape[0], size = count, replace = True)
    selected_faces = valid_faces[face_index]

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[selected_faces[:, 0]]
    tri_vectors = mesh.vertices[selected_faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis = 1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis = 1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    colors = None
    normals = None
    if w_color:
        colors = mesh.visual.vertex_colors[:, :3].astype(np.float32)
        colors = colors / 255.0
        colors = colors.view(np.ndarray)[selected_faces]
        clr_origins = colors[:, 0]
        clr_vectors = colors[:, 1:]
        clr_vectors -= np.tile(clr_origins, (1, 2)).reshape((-1, 2, 3))

        sample_color = (clr_vectors * random_lengths).sum(axis=1)
        colors = sample_color + clr_origins

        normals = mesh.face_normals[face_index]

    return samples, colors, normals


def normalize_vert_bbox(verts, attris = None, dim=-1, per_axis=False):
    bbox_min = torch.min(verts, dim=dim, keepdim=True)[0]
    bbox_max = torch.max(verts, dim=dim, keepdim=True)[0]
    if attris is not None:
        verts = attris
    verts = verts - 0.5 * (bbox_max + bbox_min)
    if per_axis:
        verts = 2 * verts / (bbox_max - bbox_min)
    else:
        verts = 2 * verts / torch.max(bbox_max-bbox_min, dim=dim, keepdim=True)[0]
    return verts
