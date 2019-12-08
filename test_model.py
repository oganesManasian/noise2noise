import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
# from noise_model import get_noise_model
# import random


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="unet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    # parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25",
    #                     help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    # parser.add_argument("--nb_tests", type=int, default=None,
    #                     help="number of files to test on")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    # nb_tests = args.nb_tests
    # val_noise_model = get_noise_model(args.test_noise_model)
    model = get_model(args.model)
    model.load_weights(weight_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    # nb_tests = min(nb_tests, len(image_paths))
    # print("Will test on {} objects".format(nb_tests))
    # for _ in range(nb_tests):
    #     image_path = random.choice(image_paths)
    for image_path in image_paths:
        # image = cv2.imread(str(image_path))
        # For images with one color channel
        image = np.expand_dims(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE), axis=2)
        h, w, _ = image.shape

        out_image = np.zeros((h, w * 2, 3), dtype=np.uint8)

        pred = model.predict(np.expand_dims(image, 0))
        denoised_image = get_image(pred[0])

        out_image[:, :w] = image
        out_image[:, w:w * 2] = denoised_image

        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
        else:
            cv2.imshow("result", out_image)
            key = cv2.waitKey(-1)
            if key == 113:  # "q": quit
                return 0


if __name__ == '__main__':
    main()
