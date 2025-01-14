# Saving images with annotations, and printing info about the different classes

import os
import cv2
import numpy as np

from tops.config import instantiate, LazyConfig
from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm

SAVE_IMAGES = False

# Used to count the total number of labels
total_labels = [0]*9
total_widths = [[] for _ in range(10)]
total_heights = [[] for _ in range(10)]
total_aspect_ratios = [[] for _ in range(10)]


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.data_train.dataloader.shuffle = False
    cfg.data_val.dataloader.shuffle = False
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def convert_boxes_coords_to_pixel_coords(boxes, width, height):
    boxes_for_first_image = boxes[0]  # This is the only image in batch
    boxes_for_first_image[:, [0, 2]] *= width
    boxes_for_first_image[:, [1, 3]] *= height
    return boxes_for_first_image.cpu().numpy()


def convert_image_to_hwc_byte(image):
    first_image_in_batch = image[0]  # This is the only image in batch
    image_pixel_values = (first_image_in_batch * 255).byte()
    image_h_w_c_format = image_pixel_values.permute(1, 2, 0)
    return image_h_w_c_format.cpu().numpy()


def visualize_boxes_on_image(batch, label_map):
    image = convert_image_to_hwc_byte(batch["image"])
    boxes = convert_boxes_coords_to_pixel_coords(batch["boxes"], batch["width"], batch["height"])
    labels = batch["labels"][0].cpu().numpy().tolist()

    for box, i in zip(boxes, labels):
        height = box[3] - box[1]
        width = box[2] - box[0]
        aspect_ratio = width/height

        total_widths[i].append(width)
        total_heights[i].append(height)
        total_aspect_ratios[i].append(aspect_ratio)

        # Added a label counter
        total_labels[i] += 1

    image_with_boxes = draw_boxes(image, boxes, labels, class_name_map=label_map)
    return image_with_boxes


def create_viz_image(batch, label_map):
    image_without_annotations = convert_image_to_hwc_byte(batch["image"])
    image_with_annotations = visualize_boxes_on_image(batch, label_map)

    # We concatinate in the height axis, so that the images are placed on top of each other
    concatinated_image = np.concatenate([
        image_without_annotations,
        image_with_annotations,
    ], axis=0)
    return concatinated_image


def create_filepath(save_folder, image_id):
    filename = "image_" + str(image_id) + ".png"
    return os.path.join(save_folder, filename)


def save_images_with_annotations(dataloader, cfg, save_folder, num_images_to_visualize, save_images):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Saving images to", save_folder)

    num_images_to_save = min(len(dataloader), num_images_to_visualize)
    dataloader = iter(dataloader)

    for i in tqdm(range(num_images_to_save)):
        batch = next(dataloader)
        viz_image = create_viz_image(batch, cfg.label_map)
        filepath = create_filepath(save_folder, i)
        if save_images:
            cv2.imwrite(filepath, viz_image[:, :, ::-1])


def print_information(num_images_to_visualize):
    """# Printing information about the annotation boxes"""

    labels_dict =               {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_percentages =        {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_width_mean =         {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_width_std =          {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_width_min =          {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_width_max =          {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_height_mean =        {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_height_std =         {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_height_min =         {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_height_max =         {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_aspect_ratio_mean =  {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}
    labels_aspect_ratio_std =   {"background": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0, "scooter": 0, "person": 0, "rider": 0}

    total_label_count = np.sum(total_labels)

    for key, labels, widths, heights, aspect_ratios in zip(labels_dict, total_labels, total_widths, total_heights, total_aspect_ratios):
        labels_dict[key] = labels
        labels_percentages[key] = str(round(labels/total_label_count*100, 0)) + ("%")
        if labels:
            labels_width_mean[key] = round(np.mean(widths), 1)
            labels_width_std[key] = round(np.std(widths), 1)
            labels_width_min[key] = round(np.min(widths), 1)
            labels_width_max[key] = round(np.max(widths), 1)
            labels_height_mean[key] = round(np.mean(heights), 1)
            labels_height_std[key] = round(np.std(heights), 1)
            labels_height_min[key] = round(np.min(heights), 1)
            labels_height_max[key] = round(np.max(heights), 1)
            labels_aspect_ratio_mean[key] = round(np.mean(aspect_ratios), 1)
            labels_aspect_ratio_std[key] = round(np.std(aspect_ratios), 1)
        else:
            labels_width_mean[key] = "No width"
            labels_width_std[key] = "No width"
            labels_width_min[key] = "No width"
            labels_width_max[key] = "No width"
            labels_height_mean[key] = "No height"
            labels_height_std[key] = "No height"
            labels_height_min[key] = "No height"
            labels_height_max[key] = "No height"
            labels_aspect_ratio_mean[key] = "No aspect ratio"
            labels_aspect_ratio_std[key] = "No aspect ratio"
        
    print()
    print("Total labels for", num_images_to_visualize, "images is:", total_label_count, "\n")
    print("Total labels per class:", labels_dict, "\n")
    print("Percentage of detected objects per class:", labels_percentages, "\n")
    print()
    print("Average width (in pixels) for each class are", labels_width_mean, "\n")
    print("The standard deviation in width (in pixels) for each class are", labels_width_std, "\n")
    print("The minimum width (in pixels) for each class are", labels_width_min, "\n")
    print("The maximum width (in pixels) for each class are", labels_width_max, "\n")
    print()
    print("Average height (in pixels) for each class are", labels_height_mean, "\n")
    print("The standard deviation in height (in pixels) for each class are", labels_height_std, "\n")
    print("The minimum height (in pixels) for each class are", labels_width_min, "\n")
    print("The maximum height (in pixels) for each class are", labels_width_max, "\n")
    print()
    print("Average aspect ratio for each class are", labels_aspect_ratio_mean, "\n")
    print("The standard deviation in aspect ratio for each class are", labels_aspect_ratio_std, "\n")


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_visualize = "train"  # or "val"
    num_images_to_visualize = 16696  # Increase this if you want to save more images

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    save_folder = os.path.join("dataset_exploration", "annotation_images")
    save_images_with_annotations(dataloader, cfg, save_folder, num_images_to_visualize, SAVE_IMAGES)

    print_information(num_images_to_visualize)


if __name__ == '__main__':
    main()
