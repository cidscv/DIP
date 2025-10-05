from dip import util
from dip import filters as fil
import cv2
import os


def run_custom_pipeline(image, output_dir, name=""):

    print(
        f"Custom Pipeline B\n{'='*20}\n\nImage = {name}\n\nChoose from the list of filters below!"
    )

    filters = {
        1: "Adjust Levels and Curves",
        2: "Apply Gamma Transform",
        3: "Apply Gaussian Filter",
        4: "Apply Median Filter",
        5: "Apply CLAHE",
        6: "High Boost Filter",
        7: "Unsharp Mask",
        8: "White Colour Balance Correction",
        9: "Exit",
    }

    while True:
        for k, v in filters.items():
            print(f"{k}: {v}")
        choice = int(input(f"Option: "))

        match choice:
            case 1:
                print(f"Choice = {filters.get(1)}")
                confirm = input(f"Is this okay? (y/n): ")
                if confirm.upper() == "Y":
                    processed = fil.adjust_levels_and_curves(image)
                    clipping = util.detect_clipping(processed)
                    for k, v in clipping.items():
                        if k == "shadows_clipped" or k == "highlights_clipped":
                            if v == True:
                                clipping_confirm = input(
                                    f"Clipping has occurred, reset? (y/n): "
                                )
                                if clipping_confirm.upper() == "Y":
                                    continue
                                else:
                                    print(f"Not resetting after clipping!")
                    image = processed
                else:
                    continue
            case 2:
                print(f"Choice = {filters.get(2)}")
                confirm = input(f"Is this okay? (y/n): ")
                if confirm.upper() == "Y":
                    processed = fil.apply_gamma_transform(image)
                    clipping = util.detect_clipping(processed)
                    for k, v in clipping.items():
                        if k == "shadows_clipped" or k == "highlights_clipped":
                            if v == True:
                                clipping_confirm = input(
                                    f"Clipping has occurred, reset? (y/n): "
                                )
                                if clipping_confirm.upper() == "Y":
                                    continue
                                else:
                                    print(f"Not resetting after clipping!")
                    image = processed
                else:
                    continue
            case 3:
                print(f"Choice = {filters.get(3)}")
                confirm = input(f"Is this okay? (y/n): ")
                if confirm.upper() == "Y":
                    processed = fil.apply_gaussian_filter(image)
                    clipping = util.detect_clipping(processed)
                    for k, v in clipping.items():
                        if k == "shadows_clipped" or k == "highlights_clipped":
                            if v == True:
                                clipping_confirm = input(
                                    f"Clipping has occurred, reset? (y/n): "
                                )
                                if clipping_confirm.upper() == "Y":
                                    continue
                                else:
                                    print(f"Not resetting after clipping!")
                    image = processed
                else:
                    continue
            case 4:
                print(f"Choice = {filters.get(4)}")
                confirm = input(f"Is this okay? (y/n): ")
                if confirm.upper() == "Y":
                    processed = fil.apply_median_filter(image)
                    clipping = util.detect_clipping(processed)
                    for k, v in clipping.items():
                        if k == "shadows_clipped" or k == "highlights_clipped":
                            if v == True:
                                clipping_confirm = input(
                                    f"Clipping has occurred, reset? (y/n): "
                                )
                                if clipping_confirm.upper() == "Y":
                                    continue
                                else:
                                    print(f"Not resetting after clipping!")
                    image = processed
                else:
                    continue
            case 5:
                print(f"Choice = {filters.get(5)}")
                confirm = input(f"Is this okay? (y/n): ")
                if confirm.upper() == "Y":
                    processed = fil.apply_clahe(image)
                    clipping = util.detect_clipping(processed)
                    for k, v in clipping.items():
                        if k == "shadows_clipped" or k == "highlights_clipped":
                            if v == True:
                                clipping_confirm = input(
                                    f"Clipping has occurred, reset? (y/n): "
                                )
                                if clipping_confirm.upper() == "Y":
                                    continue
                                else:
                                    print(f"Not resetting after clipping!")
                    image = processed
                else:
                    continue
            case 6:
                print(f"Choice = {filters.get(6)}")
                confirm = input(f"Is this okay? (y/n): ")
                if confirm.upper() == "Y":
                    processed = fil.high_boost_filter(image)
                    clipping = util.detect_clipping(processed)
                    for k, v in clipping.items():
                        if k == "shadows_clipped" or k == "highlights_clipped":
                            if v == True:
                                clipping_confirm = input(
                                    f"Clipping has occurred, reset? (y/n): "
                                )
                                if clipping_confirm.upper() == "Y":
                                    continue
                                else:
                                    print(f"Not resetting after clipping!")
                    image = processed
                else:
                    continue
            case 7:
                print(f"Choice = {filters.get(7)}")
                confirm = input(f"Is this okay? (y/n): ")
                if confirm.upper() == "Y":
                    processed = fil.unsharp_mask(image)
                    clipping = util.detect_clipping(processed)
                    for k, v in clipping.items():
                        if k == "shadows_clipped" or k == "highlights_clipped":
                            if v == True:
                                clipping_confirm = input(
                                    f"Clipping has occurred, reset? (y/n): "
                                )
                                if clipping_confirm.upper() == "Y":
                                    continue
                                else:
                                    print(f"Not resetting after clipping!")
                    image = processed
                else:
                    continue
            case 8:
                print(f"Choice = {filters.get(8)}")
                confirm = input(f"Is this okay? (y/n): ")
                if confirm.upper() == "Y":
                    processed = fil.white_balance_correction(image)
                    clipping = util.detect_clipping(processed)
                    for k, v in clipping.items():
                        if k == "shadows_clipped" or k == "highlights_clipped":
                            if v == True:
                                clipping_confirm = input(
                                    f"Clipping has occurred, reset? (y/n): "
                                )
                                if clipping_confirm.upper() == "Y":
                                    continue
                                else:
                                    print(f"Not resetting after clipping!")
                    image = processed
                else:
                    continue
            case 9:
                print(f"Exiting. Thank you!")
                break
            case _:
                print("Invalid Entry - Try Again!")
                continue

    processed = image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Output path should be '<image_name>_<pipeline>_processed.jpg'
    output_path = os.path.join(output_dir, f"{name}_custom_processed.jpg")
    cv2.imwrite(output_path, processed)
    print(f"Processed image saved to: {output_path}")

    return processed
