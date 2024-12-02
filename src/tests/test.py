import os
import cv2
from src.localizator import localizator
import json
import numpy as np


class ViaImgMetadataParser:
    def __init__(self, json_file, dir_imgs):
        self.json_data = json.load(open(json_file, "r"))
        self.img_metadata = self.json_data["_via_img_metadata"]
        self.dir_imgs = dir_imgs
        self.polygons = self.load_polygons()

    def load_polygons(self):
        polygons = []
        for _, img_data in self.img_metadata.items():
            regions = img_data["regions"]
            for region in regions:
                if region["shape_attributes"]["name"] == "polygon":
                    all_points_x = region["shape_attributes"]["all_points_x"]
                    all_points_y = region["shape_attributes"]["all_points_y"]
                    polygon_points = list(zip(all_points_x, all_points_y))
                    polygons.append(
                        {
                            "polygon": polygon_points,
                            "img": os.path.join(self.dir_imgs, img_data["filename"]),
                        }
                    )
        return polygons

    def __iter__(self):
        return ViaPolygonsIterator(self.polygons)


class ViaPolygonsIterator:
    def __init__(self, polygons):
        self.polygons = polygons
        self.index = 0

    def __next__(self):
        if self.index < len(self.polygons):
            img_id = self.polygons[self.index]["img"]
            polygon_points = self.polygons[self.index]["polygon"]
            self.index += 1
            return img_id, polygon_points
        else:
            raise StopIteration


def create_box_image(image, polygon):
    x, y, w, h = cv2.boundingRect(np.array([polygon], dtype=np.int32))
    box_image = np.zeros((h, w, 3), dtype=np.uint8)

    box_image[:, :] = image[y : y + h, x : x + w]

    relative_polygon = polygon - np.array([x, y])

    return box_image, relative_polygon


def calculate_iou(polygon1, polygon2, image):
    if len(polygon1) == 0 or len(polygon2) == 0:
        return 0
    height, width, _ = image.shape

    mask1 = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask1, [polygon1], -1, 255, -1)

    mask2 = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask2, [polygon2], -1, 255, -1)

    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    iou = np.sum(intersection) / np.sum(union)

    return iou


def test_detect_object_and_calculate_iou(conf: str, dir_imgs: str, output: str):
    loc = localizator.Localizator()

    curr_image = None
    curr_index = 0

    for img_path, polygon_points in ViaImgMetadataParser(conf, dir_imgs):
        if curr_image != img_path:
            curr_image = img_path
            curr_index = 0

        file_name = os.path.basename(img_path)
        dir_name = f"ind_{curr_index}_img_{os.path.splitext(file_name)[0]}"

        if not os.path.exists(os.path.join(output, dir_name)):
            os.makedirs(os.path.join(output, dir_name))

        res_img = cv2.imread(img_path)

        box_image, relative_polygon = create_box_image(res_img, polygon_points)

        points, gradient, blurred, thresh, closed = loc.localize_test(box_image)

        assert points is not None

        iou = calculate_iou(relative_polygon, points, box_image)
        cv2.imwrite(os.path.join(output, dir_name, "start.jpg"), box_image)

        cv2.drawContours(box_image, np.array([relative_polygon], dtype=np.int32), -1, (0, 0, 255), 1)
        cv2.drawContours(box_image, np.array([points], dtype=np.int32), -1, (0, 255, 0), 1)
        cv2.putText(box_image, f"IoU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output, dir_name, "img.jpg"), box_image)
        cv2.imwrite(os.path.join(output, dir_name, "gradient.jpg"), gradient)
        cv2.imwrite(os.path.join(output, dir_name, "blurred.jpg"), blurred)
        cv2.imwrite(os.path.join(output, dir_name, "thresh.jpg"), thresh)
        cv2.imwrite(os.path.join(output, dir_name, "closed.jpg"), closed)

        curr_index += 1
