import glob, os
from skimage.transform import resize
import cv2 as cv
import numpy as np

def load_image(path):
  x = np.array(cv.imread(path))
  x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
  return resize(x, (224,224,3))

def load_data(root_dir = './Dataset/'):
    data = []
    for path, subdirs, files in os.walk(root_dir):
      for user in subdirs:
        v=[]
        for filename in glob.iglob(root_dir + user + '**/*.jpg', recursive=True):
          v.append(load_image(filename))

        v=np.array(v)
        data.append(v)

    data = np.array(data)
    return data