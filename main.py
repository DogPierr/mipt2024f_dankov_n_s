import numpy as np
import argparse
import cv2
from src.localizator import localizator


def main():
    """
    Main function to localize barcodes and QR codes in an image and save the output.

    This function parses command-line arguments to get the input image path and optional
    output path. It reads the image, uses the Localizator class to detect barcodes and
    QR codes, draws the detected contours on the image, and saves the annotated image
    to the specified output path or to 'res.png' by default.

    Command-line arguments:
    - -i, --image: Required. The path to the input image.
    - -o, --output: Optional. The path to save the output image. Defaults to 'res.png' if not provided.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Image path", required=True)
    parser.add_argument("-o", "--output", type=str, help="Ouput path", required=False)

    args = parser.parse_args()

    img_path = args.image
    output_path = args.output

    res_img = cv2.imread(img_path)

    loc = localizator.Localizator()

    points = loc.localize(res_img)

    if points is not None:
        for p in points:
            # print(np.array(p, dtype=np.int32))
            res_img = cv2.polylines(res_img, [np.array(p, dtype=np.int32)], True, (0, 255, 0), 3)

    if output_path is not None:
        cv2.imwrite(output_path, res_img)
    else:
        cv2.imwrite("res.png", res_img)


if __name__ == "__main__":
    main()
