import numpy
from cv2.cv2 import *

image = imread('fog.jpg')
imshow("Original", image)
gray = cvtColor(image, COLOR_BGR2GRAY)
imshow("Gray", gray)
eq = equalizeHist(gray)
imshow("Histogram Equalization", numpy.hstack([gray, eq]))
imwrite("filter.jpg", numpy.hstack([gray, eq]))
if waitKey(0) == 27:
    destroyAllWindows()
