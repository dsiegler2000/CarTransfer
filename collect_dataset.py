import cv2
import os

BLUR_KERNEL = (3, 3)
DDEPTH = cv2.CV_16S
KSIZE = 3
SCALE = 1
DELTA = 0
TARGET_SIZE = (512, 512)
MODE = "resize"

i = 0

if MODE == "generate_sketch":
    for f in os.listdir("raw_data"):
        try:
            im = cv2.imread(f"raw_data/{f}")

            # Blur and grayscale
            blur = cv2.GaussianBlur(im, (3, 3), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            # Compute gradients
            grad_x = cv2.Sobel(gray, DDEPTH, 1, 0, ksize=KSIZE, scale=KSIZE, delta=DELTA, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(gray, DDEPTH, 0, 1, ksize=KSIZE, scale=KSIZE, delta=DELTA, borderType=cv2.BORDER_DEFAULT)

            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            grad = cv2.resize(grad, TARGET_SIZE)
            im = cv2.resize(im, TARGET_SIZE)

            cv2.imwrite(f"sketch/{f}", grad)
            cv2.imwrite(f"color/{f}", im)
            i += 1
            if i % 500 == 0:
                print(f"{i}/{len(os.listdir('raw_data'))}")
        except Exception as e:
            print(e)
            print(f)
elif MODE == "resize":
    def resize(filepath):
        return cv2.resize(cv2.imread(filepath), TARGET_SIZE)
    filelist = os.listdir("sketch") + os.listdir("color")
    n = len(filelist)
    for f in filelist:
        f = os.path.join("sketch" if i / n < .5 else "color", f)
        im = resize(f)
        cv2.imwrite(f, im)
        i += 1

        if i % 500 == 0:
            print(f"{i} / {n}")

