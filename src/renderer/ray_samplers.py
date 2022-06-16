"""
Ray samplers for sampling rays used for volume rendering.
"""


class RaySamplerBase(object):
    """
    Base class for ray samplers.
    """

    def __init__(self):
        pass

    def sample_rays(self, *args, **kwargs):
        """
        Sample rays.
        """
        raise NotImplementedError()
