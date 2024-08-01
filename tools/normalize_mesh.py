import numpy as np

def normalize_mesh(mesh):
    """
    Normalize the mesh to [-0.5, 0.5] range.
    """
    # normalize the points
    bbmin, bbmax = np.min(mesh, axis=0), np.max(mesh, axis=0)
    center = (bbmin + bbmax) / 2.0
    radius = 2.0 / (np.max(bbmax - bbmin) + 1.0e-6)
    normalized_mesh = 0.5 * (mesh - center) * radius  # normalize to [-0.5, 0.5]
    return normalized_mesh

def read_off(file_path):
    """
    Read a mesh in the .off format.
    """
    with open(file_path, 'r') as f:
        # Read the header.
        header = f.readline().strip()
        if header != 'OFF':
            raise ValueError('Invalid .off file format.')

        # Read the number of vertices, faces, and edges.
        num_vertices, num_faces, num_edges = map(int, f.readline().strip().split())

        # Read the vertices.
        vertices = []
        for i in range(num_vertices):
            vertex = list(map(float, f.readline().strip().split()))
            vertices.append(vertex)

        # Read the faces.
        faces = []
        for i in range(num_faces):
            face = list(map(int, f.readline().strip().split()))[1:]
            faces.append(face)

    # Convert the vertices and faces to numpy arrays.
    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices, faces

def write_off(file_path, vertices, faces):
    """
    Write a mesh in the .off format.
    """
    with open(file_path, 'w') as f:
        # Write the header.
        f.write('OFF\n')

        # Write the number of vertices, faces, and edges.
        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]
        num_edges = 0
        f.write(f'{num_vertices} {num_faces} {num_edges}\n')

        # Write the vertices.
        for i in range(num_vertices):
            vertex = vertices[i]
            f.write(f'{vertex[0]} {vertex[1]} {vertex[2]}\n')

        # Write the faces.
        for i in range(num_faces):
            face = faces[i]
            f.write(f'{len(face)} {" ".join(str(v) for v in face)}\n')

def normalize_off(file_path, save_folder, save_name):
    """
    Normalize a mesh in the .off format to [-0.5, 0.5] range.
    """
    # Read the mesh.
    vertices, faces = read_off(file_path)

    # Normalize the vertices.
    normalized_vertices = normalize_mesh(vertices)

    # Write the normalized mesh to a new file.
    write_off(save_folder + '/' + save_name, normalized_vertices, faces)
