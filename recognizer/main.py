# import the necessary packages
from PIL import Image
import pytesseract
import argparse
from cv2 import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to input file, using device camera by default")
ap.add_argument("-m", "--mode", type=str, default="ocr", help="work mode, <line | circle | ocr>, ocr by default")
ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
args = vars(ap.parse_args())
file_path = args["file"]
# load the example image and convert it to grayscale
video = cv2.VideoCapture(0) if file_path is None else cv2.VideoCapture(file_path)

while True:
    ret, frame = video.read()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if args["mode"] == "line":
        output = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([-10, 43, 46])
        upper_red = np.array([10, 255, 255])
        red_line = cv2.inRange(hsv, lower_red, upper_red)
        xs = lambda y: [x for x in range(len(red_line[y])) if red_line[y][x] > 0]
        y_start = 0
        y_end = len(red_line) - 1
        xs_start = xs(y_start)
        xs_end = xs(y_end)
        if len(xs_start) != 0 and len(xs_end) != 0:
            x_start = int(np.mean(xs_start))
            x_end = int(np.mean(xs_end))
            # print("start: (%d, %d), end: (%d, %d)" %(x_start, y_start, x_end, y_end))
            cv2.line(output, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
        cv2.imshow("output", np.hstack([frame, output]))
    if args["mode"] == "circle":
        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
        # ensure at least some circles were found
        if circles is not None:
            output = frame.copy()
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            cv2.imshow("output", np.hstack([frame, output]))

    if args["mode"] == "ocr":
        cv2.imshow('frame', frame)
        # check to see if we should apply thresholding to preprocess the
        # image
        if args["preprocess"] == "thresh":
            gray = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # make a check to see if median blurring should be done to remove
        # noise
        elif args["preprocess"] == "blur":
            gray = cv2.medianBlur(gray, 3)
        cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)
        cv2.imwrite("output/ocr.png", gray)
        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        # filename = os.path.basename(args["image"])
        # file_path = os.path.join("output", filename)
        # cv2.imwrite(file_path, gray)
        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        text = pytesseract.image_to_string(Image.open("output/ocr.png"))
        if text != "":
            print(text)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
