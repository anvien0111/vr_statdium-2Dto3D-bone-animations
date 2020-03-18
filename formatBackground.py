import sys
import cv2
import os
from sys import platform
import argparse
import json
import math
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
c  = cv2.VideoCapture(0)
_, f = c.read()

avg1 = np.float32(f)
avg2 = np.float32(f)

while (1):
    _, f = c.read()

   # cv2.accumulateWeighted(f, avg1, 0.1)
    cv2.accumulateWeighted(f, avg2, 0.001)

  #  res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    cv2.imshow('img', f)
   # cv2.imshow('avg1', res1)
    cv2.imshow('avg2', res2)
    k = cv2.waitKey(20)

    if k == 27:
        break

cv2.destroyAllWindows()
c.release()

