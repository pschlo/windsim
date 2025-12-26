from typing import Any, Literal, cast

import numpy as np
import pyrender
import trimesh
from trimesh import Trimesh
from trimesh.path import Path2D, Path3D


def path_to_mesh(path: Path2D | Path3D, radius: float = 0.5, sections: int = 4) -> Trimesh:
    cylinders = []
    for polyline in path.entities:
        polyline_segments = path.vertices[polyline.nodes]  # 3d (n X 2 X 3)
        for segment in polyline_segments:
            assert isinstance(segment, np.ndarray)
            if segment.shape[1] == 2:
                segment = np.column_stack((segment, np.zeros(2)))
            cylinders.append(trimesh.creation.cylinder(radius=radius, sections=sections, segment=segment))
    return cast(Trimesh, trimesh.boolean.union(cylinders))

def point_to_mesh(point, radius=1.0, color=None) -> Trimesh:
    mesh = trimesh.creation.icosphere(radius=radius).apply_translation(point)
    if color is not None:
        mesh.visual.vertex_colors = color
    return mesh


def render_3d(barriers: Trimesh, s = None, r = None, *, top_path: Path3D | None = None, left_path: Path3D | None = None, right_path: Path3D | None = None, renderer: Literal['trimesh', 'pyrender'], lines_to_mesh: bool = True):
    # remesh
    # barriers = barriers.subdivide().subdivide()
    
    if top_path is not None:
        # cut barriers to relevant part
        BUFFER = 500
        bounding_box = trimesh.creation.box(bounds=top_path.bounds + [(-BUFFER, -BUFFER, -BUFFER), (BUFFER, BUFFER, BUFFER)])
        clipped_barriers = barriers.slice_plane(bounding_box.facets_origin, -bounding_box.facets_normal)
        assert isinstance(clipped_barriers, Trimesh)
    else:
        bounding_box = barriers.bounding_box
        clipped_barriers = barriers

    s_mesh, r_mesh = None, None
    if s is not None and r is not None:
        s_mesh = point_to_mesh(s, radius=8, color=[0,255,0])
        r_mesh = point_to_mesh(r, radius=2, color=[255,0,0])

    # move everything towards origin
    translation = np.hstack([-bounding_box.center_mass[:2], 0])
    clipped_barriers.apply_translation(translation)
    if top_path is not None:
        top_path.apply_translation(translation)
    if left_path is not None:
        left_path.apply_translation(translation)
    if right_path is not None:
        right_path.apply_translation(translation)
    if s_mesh is not None:
        s_mesh.apply_translation(translation)
    if r_mesh is not None:
        r_mesh.apply_translation(translation)

    # set barrier color
    # clipped_barriers.visual.vertex_colors = [133,133,133,255]  # type: ignore
    # clipped_barriers.visual.vertex_colors = [180,180,180,255]  # type: ignore
    clipped_barriers.visual.vertex_colors = [255,255,255,255]  # type: ignore

    # define scene
    scene = trimesh.Scene([clipped_barriers])

    # add ring 1
    if top_path is not None:
        if lines_to_mesh:
            ring_mesh = path_to_mesh(top_path)
            ring_mesh.visual.vertex_colors = [255,255,0]  # type: ignore
            scene.add_geometry(ring_mesh)
        else:
            top_path.colors = [[0, 0, 255]] * len(top_path.entities)
            scene.add_geometry(top_path)

    # add ring 2
    if left_path is not None:
        if lines_to_mesh:
            ring_mesh = path_to_mesh(left_path)
            ring_mesh.visual.vertex_colors = [0,0,255]  # type: ignore
            scene.add_geometry(ring_mesh)
        else:
            left_path.colors = [[0, 0, 255]] * len(left_path.entities)
            scene.add_geometry(left_path)

    # add ring 2
    if right_path is not None:
        if lines_to_mesh:
            ring_mesh = path_to_mesh(right_path)
            ring_mesh.visual.vertex_colors = [0,0,255]  # type: ignore
            scene.add_geometry(ring_mesh)
        else:
            right_path.colors = [[0, 0, 255]] * len(right_path.entities)
            scene.add_geometry(right_path)



    
    # add ground plane
    PLANE_HEIGHT = 2
    ground_plane = trimesh.creation.box([2000, 2000, PLANE_HEIGHT], trimesh.transformations.translation_matrix([0,0,-PLANE_HEIGHT/2]))
    # ground_plane.visual.vertex_colors = [255,255,255]

    from PIL import Image, ImageDraw
    from trimesh.visual.material import SimpleMaterial
    from trimesh.visual.texture import TextureVisuals

    # Step 1: Create a plane mesh with small thickness
    plane_mesh = ground_plane

    # Step 2: Generate a high-resolution grid image using NumPy
    def create_grid_image(width, height, grid_size, line_width=1, line_color=(0, 0, 0), bg_color=(255, 255, 255)):
        # Create a background image
        image_array = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Draw vertical grid lines
        for x in range(0, width, grid_size):
            image_array[:, x:x+line_width] = line_color
        
        # Draw horizontal grid lines
        for y in range(0, height, grid_size):
            image_array[y:y+line_width, :] = line_color
        
        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(image_array, 'RGB')
        return image

    # Increase the resolution for sharper lines
    grid_image = create_grid_image(
        width=5_000,  # Higher resolution width
        height=5_000,  # Higher resolution height
        grid_size=32,  # Adjust grid size to match the new resolution
        line_width=1,  # Set line width to make lines thinner or thicker
        line_color=(0, 0, 0),
        # bg_color=(250, 250, 250)
        bg_color=(150, 150, 150)
    )

    # Step 3: Compute UV coordinates for the mesh vertices
    # Extract the vertex positions
    vertices = plane_mesh.vertices
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Normalize x and y to range [0, 1] for UV mapping
    u = (x - x.min()) / (x.max() - x.min())
    v = (y - y.min()) / (y.max() - y.min())
    uv = np.column_stack((u, v))

    material = SimpleMaterial(image=grid_image)
    plane_mesh.visual = TextureVisuals(uv=uv, material=material)


    scene.add_geometry(ground_plane)

    # add points
    if s_mesh is not None:
        scene.add_geometry(s_mesh)
    if r_mesh is not None:
        scene.add_geometry(r_mesh)

    # add origin
    # origin = trimesh.creation.icosphere(radius=1)
    # origin.visual.vertex_colors = [255,0,0]
    # scene.add_geometry(origin)



    # render
    if renderer == "trimesh":
        scene.camera.z_far = 1_000_000_000
        import pyglet
        config = pyglet.gl.Config(
            sample_buffers=1,  # Enable sample buffers
            samples=8,         # Number of samples for anti-aliasing
            depth_size=24,     # Depth buffer size
            double_buffer=True # Enable double buffering
        )


        scene.show(window_conf=config, smooth=False, resolution=(1920, 1080))
    elif renderer == "pyrender":
        import pyrender  # type: ignore

        # Assuming you have a trimesh scene `scene`
        pyrender_scene = from_trimesh_scene(scene)#, ambient_light=[0.02, 0.02, 0.02])

        # Calculate the center of the scene's bounding box
        bounding_box_center = scene.bounds.mean(axis=0)

        # Set up the camera above the center of the bounding box
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, znear=0.01, zfar=10000000.0)

        assert r is not None
        s = np.sqrt(2)/2
        camera_pose = np.array([
            [0.0, -s,   s,   r_mesh.center_mass[0]],
            [1.0,  0.0, 0.0, r_mesh.center_mass[1]],
            [0.0,  s,   s,   2],
            [0.0,  0.0, 0.0, 1.0],
        ])

        camera_pose_1 = np.array([[ 2.02859812e-01, -4.57774469e-01,  8.65615638e-01,  4.08660076e+02],
 [ 9.79193601e-01,  9.00756742e-02, -1.81841315e-01, -1.54133454e+02],
 [ 5.27139920e-03,  8.84493589e-01,  4.66522565e-01,  6.34892874e+01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        my_node = pyrender.Node(matrix=camera_pose_1)
        pyrender_scene.add_node(my_node)
        

        camera_pose = np.array([[ 3.87483099e-01, -3.55584892e-01,  8.50538790e-01,  4.00120061e+02],
 [ 9.21838161e-01,  1.41005196e-01, -3.61015151e-01, -1.45289203e+02],
 [ 8.44114472e-03,  9.23946384e-01,  3.82428853e-01,  3.37833411e+01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        # # Create a camera pose matrix
        # camera_pose = np.eye(4)

        # # Position the camera above the scene
        # camera_pose[0:3, 3] = [bounding_box_center[0], bounding_box_center[1], bounding_box_center[2] + 10]  # 10 units above

        # # Orient the camera to look directly downwards (negative Z-axis)
        # camera_pose[0:3, 2] = [0, 0, 1]  # Pointing downwards
        # camera_pose[0:3, 1] = [0, 1, 0]   # Y-axis pointing up
        # camera_pose[0:3, 0] = [1, 0, 0]  # X-axis pointing left

        # Add the camera to the pyrender scene
        pyrender_scene.add(camera, pose=camera_pose_1)

        add_raymond_lights(pyrender_scene, intensity=0.3)



        # Primary directional light for strong shadows
        # light_1 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        # pyrender_scene.add(light_1, pose=camera_pose)


        light_2 = pyrender.DirectionalLight(color=np.ones(3), intensity=2)
        direction_2 = np.array([0.5, -0.5, -1], dtype=np.float64)
        pyrender_scene.add(
            light_2,
            pose = trimesh.geometry.align_vectors([0, 0, -1], direction_2 / np.linalg.norm(direction_2))
        )


        # light_2 = pyrender.DirectionalLight(color=np.ones(3), intensity=2)
        # direction_2 = np.array([-0.5, -0.5, -1], dtype=np.float64)
        # pyrender_scene.add(
        #     light_2,
        #     pose = trimesh.geometry.align_vectors([0, 0, -1], direction_2 / np.linalg.norm(direction_2))
        # )


        # light_3 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        # direction_3 = np.array([0.5, 0.5, -1], dtype=np.float64)
        # pyrender_scene.add(
        #     light_3,
        #     pose = trimesh.geometry.align_vectors([0, 0, -1], direction_3 / np.linalg.norm(direction_3))
        # )

        # light_4 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        # direction_4 = np.array([-0.5, 0.5, -1], dtype=np.float64)
        # pyrender_scene.add(
        #     light_4,
        #     pose = trimesh.geometry.align_vectors([0, 0, -1], direction_4 / np.linalg.norm(direction_4))
        # )

        # light_5 = pyrender.DirectionalLight(color=np.ones(3), intensity=0.1)
        # direction_5 = np.array([-1, -1, -1], dtype=np.float64)
        # pyrender_scene.add(
        #     light_5,
        #     pose = trimesh.geometry.align_vectors([0, 0, -1], direction_5 / np.linalg.norm(direction_5))
        # )

        # viewer = pyrender.Viewer(
        #     pyrender_scene,
        #     fullscreen=True
        # )
        # matrix = viewer._camera_node.matrix
        # print(matrix)
        # exit()

        # Render the scene using pyrender.Viewer
        r = pyrender.OffscreenRenderer(1920 * 2, 1080 * 2)  # Render at 2x resolution for sharper edges

        # Render the scene with anti-aliasing (MSAA)
        render_flags = pyrender.RenderFlags.SHADOWS_ALL
        color, depth = r.render(pyrender_scene, flags=render_flags)

        # Optionally downscale the image if needed
        from PIL import Image
        image = Image.fromarray(color)
        # image = image.resize((1024, 768), Image.ANTIALIAS)
        image.show()
        exit()





def from_trimesh_scene(trimesh_scene,
                           bg_color=None, ambient_light=None):
        """Create a :class:`.Scene` from a :class:`trimesh.scene.scene.Scene`.

        Parameters
        ----------
        trimesh_scene : :class:`trimesh.scene.scene.Scene`
            Scene with :class:~`trimesh.base.Trimesh` objects.
        bg_color : (4,) float
            Background color for the created scene.
        ambient_light : (3,) float or None
            Ambient light in the scene.

        Returns
        -------
        scene_pr : :class:`Scene`
            A scene containing the same geometry as the trimesh scene.
        """
        import pyrender

        material = pyrender.MetallicRoughnessMaterial(
            # baseColorFactor=[1.0, 0.5, 0.5, 1.0],  # Red color
            metallicFactor=0.0,  # Non-metallic
            roughnessFactor=0.1  # High roughness to diffuse light
        )

        # convert trimesh geometries to pyrender geometries
        geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                      for name, geom in trimesh_scene.geometry.items()}

        # create the pyrender scene object
        scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)

        # add every node with geometry to the pyrender scene
        for node in trimesh_scene.graph.nodes_geometry:
            pose, geom_name = trimesh_scene.graph[node]
            scene_pr.add(geometries[geom_name], pose=pose)

        return scene_pr



def add_raymond_lights(scene: pyrender.Scene, intensity=1.0, parent: pyrender.Node | None = None):
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]

        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
            pose=matrix,
            parent_node=parent if parent is not None else next(iter(scene.camera_nodes))
        )