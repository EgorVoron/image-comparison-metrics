import pdqhash
import imagehash
import numpy as np
from PIL import Image


class PDQ_Hash(imagehash.ImageHash):
    # for future
    def __hash__(self):
        score, ac = pdqhash.compute(np.array(self.hash.flatten()))
        return imagehash.ImageHash(score)


def pdq_hash(image: Image.Image):
    return imagehash.ImageHash(pdqhash.compute(np.array(image))[0])
