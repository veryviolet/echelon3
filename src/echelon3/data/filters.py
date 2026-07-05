from abc import abstractmethod
from PIL import Image

from echelon3.utils.get_image_size import get_image_size

ORIENTATION_PORTRAIT = 'portrait'
ORIENTATION_LANDSCAPE = 'landscape'

ORIENTATIONS = [ORIENTATION_PORTRAIT, ORIENTATION_LANDSCAPE]

class BaseFilter:

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def check_file(self, filename, **kwargs):
        pass


class OrientationFilter:

    orientation = None

    def __init__(self, orientation, **kwargs):
        if orientation not in ORIENTATIONS:
            raise RuntimeError(f'orientation should be one of: {ORIENTATIONS}')

        self.orientation = orientation

    def check_file(self, filename, **kwargs):
        width, height = get_image_size(filename)
#        im = Image.open(filename)
#        width, height = im.size
#        im.close()

        return width <= height if self.orientation == ORIENTATION_PORTRAIT else width > height


