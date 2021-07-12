"""
Volume renderer for NeRF.
"""


class Renderer:
    def __init__(
        self,
        scene_repr,
        camera_orig,
        z_near=0.5,
        z_far=7,
        img_size=(800, 800),
        num_samples=100,
        use_hierarchical=True,
    ):
        """
        Volume renderer for NeRF.
        Computes pixel value by casting rays to neural radiance fields.

        Args:
        - scene_repr (nn.Module): Neural representation of a scene to be rendered.
        - camera_orig (tuple): 3-tuple containing (x, y, z) of camera in world space.
        - z_near (float): Nearest visible depth.
        - z_far (float): Farthest visible depth.
        - img_size (tuple): 2-tuple containing width and height of render output dimension.
        - num_samples (int): Number of samples used to compute a pixel value.
        - use_hierarchical (boolean): Use hierarchical sampling or not.
        """

        self.scene_repr = scene_repr
        self.camera_orig = camera_orig
        self.z_near = z_near
        self.z_far = z_far
        self.img_width = img_size[0]
        self.img_height = img_size[1]
        self.num_samples = num_samples
        self.use_hierarchical = use_hierarchical

    def render(self):
        """
        Render the scene.

        Args:
        - TBD

        Returns:
        - TBD
        """

        for w_idx in range(self.img_width):
            for h_idx in range(self.img_height):
                # compute pixel value
                pass

        # return rendered image

    def sample_points(self):
        """
        Sample points along a ray.

        Args:
        - TBD

        Returns:
        - TBD
        """

    def cast_ray(self, ray_dir):
        """
        Cast ray from the camera position to a given direction.

        Args:
        - ray_dir (tuple): 3-tuple designating ray direction.

        Returns:
        - TBD
        """
        pass
