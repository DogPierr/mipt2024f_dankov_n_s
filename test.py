import argparse
from src.tests import test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, help="Config path", required=True)
    parser.add_argument("-d", "--data", type=str, help="Data path", required=True)
    parser.add_argument("-o", "--output", type=str, help="Ouput path", required=False)

    args = parser.parse_args()

    conf_path = args.conf
    data_path = args.data
    output_path = args.output

    test.test_detect_object_and_calculate_iou(conf_path, data_path, output_path)


if __name__ == "__main__":
    main()
