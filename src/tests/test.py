import os
import cv2
from src.localizator import localizator
import json
import numpy as np

CONF = ""
TEST_GLOB = ""
DIR_IMGS = ""


class ViaImgMetadataParser:
    def __init__(self, json_file):
        self.json_data = json.load(open(json_file, "r"))
        self.img_metadata = self.json_data["_via_img_metadata"]
        self.polygons = self.load_polygons()

    def load_polygons(self):
        polygons = []
        for img_id, img_data in self.img_metadata.items():
            regions = img_data["regions"]
            for region in regions:
                if region["shape_attributes"]["name"] == "polygon":
                    all_points_x = region["shape_attributes"]["all_points_x"]
                    all_points_y = region["shape_attributes"]["all_points_y"]
                    polygon_points = list(zip(all_points_x, all_points_y))
                    polygons = {
                        "polygon": [polygon_points],
                        "img": os.path.join(DIR_IMGS, img_id),
                    }
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
    # Find the bounding box of the polygon
    x, y, w, h = cv2.boundingRect(polygon)

    # Create a black image with the same size as the bounding box
    box_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Copy the region of the original image containing the polygon to the box image
    box_image[:, :] = image[y : y + h, x : x + w]

    # Find the coordinates of the polygon relative to the new smaller image
    relative_polygon = polygon - np.array([x, y])

    # Return the box image and the coordinates of the polygon relative to the box image
    return box_image, relative_polygon


def calculate_iou(polygon1, polygon2):
    # Create a mask for the first polygon
    mask1 = np.zeros((512, 512), dtype=np.uint8)
    cv2.drawContours(mask1, [polygon1], -1, 255, -1)

    # Create a mask for the second polygon
    mask2 = np.zeros((512, 512), dtype=np.uint8)
    cv2.drawContours(mask2, [polygon2], -1, 255, -1)

    # Calculate the intersection
    intersection = cv2.bitwise_and(mask1, mask2)

    # Calculate the union
    union = cv2.bitwise_or(mask1, mask2)

    # Calculate the IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou


def test_detect_object_and_calculate_iou():
    loc = localizator.Localizator()

    for img_path, polygon_points in ViaImgMetadataParser(CONF):
        res_img = cv2.imread(img_path)

        box_image, relative_polygon = create_box_image(res_img, polygon_points)

        points = loc.localize(box_image)

        assert points is not None

        for p in points:
            assert p in polygon_points
