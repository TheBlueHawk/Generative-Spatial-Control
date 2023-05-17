import datetime
from typing import Dict, List, Tuple
import json
import random
import randomname
import os
import shutil
import tqdm

from PIL import Image
import torch
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor, OwlViTProcessor, OwlViTForObjectDetection

import data_types
import script

# dump 80 MSCOCO object categories here for quick reference
categories = 'person, bicycle, car, motorcycle, airplane, ' \
             'bus, train, truck, boat, traffic light, ' \
             'fire hydrant, stop sign, parking meter, ' \
             'bench, bird, cat, dog, horse, sheep, cow, ' \
             'elephant, bear, zebra, giraffe, backpack, ' \
             'umbrella, handbag, tie, suitcase, frisbee, ' \
             'skis, snowboard, sports ball, kite, ' \
             'baseball bat, baseball glove, skateboard, ' \
             'surfboard, tennis racket, bottle, wine' \
             'glass, cup, fork, knife, spoon, bowl, ' \
             'banana, apple, sandwich, orange, broccoli, ' \
             'carrot, hot dog, pizza, donut, cake, chair, ' \
             'couch, potted plant, bed, dining table, ' \
             'toilet, tv, laptop, mouse, remote, keyboard, ' \
             'cell phone, microwave, oven, toaster, sink, ' \
             'refrigerator, book, clock, vase, scissors, ' \
             'teddy bear, hair drier, toothbrush'.split(", ")

super_categories = 'person, vehicle, outdoor, animal, accessory, ' \
                   'sports, kitchen, food, furniture, ' \
                   'electronic, appliance, indoor'.split(", ")

# TODO: we need to create a list of objects that belong to different categories
#       desirably with completely different contexts

def _generate_prompt(category_a, category_b):
    """
        takes two categories and returns a prompt that includes spatial relation between them
        the prompt is of the form "A/An <category_a> is <rel> a/an <category b>"
    """
    relationships = [
        ("left", "to the left of"),
        ("above", "above"),
        ("below", "below"),
        ("right", "to the right of"),
    ]
    relationship, rel_str = random.choice(relationships)

    article_a = "An" if category_a[0].lower() in "aeiou" else "A" # h
    article_b = "an" if category_b[0].lower() in "aeiou" else "a"

    prompt = f"{article_a} {category_a} is {rel_str} {article_b} {category_b}"
    return relationship, prompt


def generate_all_prompts(categories, with_self=False, experiment="baseline_exp"):
    """
        generates prompts using a list of categories
        with_self: whether to use self-pairing
    """
    prompts = []
    for i in range(len(categories)):
        for j in range(i + with_self, len(categories)):
            category_a = categories[i]
            category_b = categories[j]
            rel, prompt = _generate_prompt(category_a, category_b, experiment=experiment)
            prompts.append((category_a, category_b, prompt, rel))
    return prompts


def get_random_prompt(categories=categories, experiment="baseline_exp") -> data_types.VisorTuple:
    """Get a random prompt. Self-pairing is not allowed."""
    category_a = random.choice(categories)
    category_b = random.choice(categories)
    while category_a == category_b:
        category_b = random.choice(categories)
    relationship, prompt = _generate_prompt(category_a, category_b)
    return data_types.VisorTuple(category_a, category_b, prompt, relationship)


def dump_prompts_to_file(prompts, file_name):
    """
        dumps generated prompt to file
    """
    with open(file_name, "w") as file:
        for prompt in prompts:
            file.write(prompt + "\n")


def visualize_bounding_boxes(coords, scores, labels, img, out_path: pathlib.Path = None):
    """
        given a list of box coordinates, confidence scores, labels and a PIL image {img},
        compute the centroids of the boxes
        visualize the bounding boxes and the centroids
    """

    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Convert the coordinates to numpy array
    coords = np.array(coords)

    # Visualize the bounding boxes
    for bbox, score, label in zip(coords, scores, labels):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        height, width =  abs(y2-y1), abs(x2-x1)
        # x = x1 if x1 > x2 else x2
        # y = y1 if y1 > y2 else y2
        box = plt.Rectangle((x1, y1), height=height, width=width, fill=False, edgecolor='red', lw=3)
        ax.add_patch(box)

        # Calculate centroid coordinates
        centroid_x = x1 + (width / 2)
        centroid_y = y1 + (height / 2)

        # Visualize the centroids
        ax.plot(centroid_x, centroid_y, marker='o', markersize=3, color='red')

        # Show label and confidence score
        rx, ry = box.get_xy()
        cx = rx + box.get_width() / 2.0
        cy = ry + box.get_height() / 8.0
        l = ax.annotate(
            f"{label}: {score}",
            (cx, cy),
            fontsize=8,
            fontweight="bold",
            color="white",
            ha='center',
            va='center'
        )
        l.set_bbox(
            dict(facecolor='red', alpha=0.5, edgecolor='red')
        )

    # Show the image with bounding boxes and centroids
    if out_path is not None:
        out_path = pathlib.Path(out_path)
        out_path = out_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path))

    plt.show()

# Computes OA score
def compute_OA_score(text, labels):
    # no need to check for placement, just the presence of all objects
    if len(set(labels)) == len(text):
        return 1
    else:
        return 0

# Computes VISOR score
def compute_VISOR_score(text, image_size, coords, scores, labels, experiment="baseline_exp"):
    # if baseline experiment, no need to check for placement, just the presence of all objects
    if experiment == "baseline_exp":
        if len(set(labels)) == len(text):
            return 1
        else:
            return 0
    else:
        assert len(text) == 2 # Code only works for two objects

        # Convert to numpy array
        coords = np.array(coords)
        image_size = np.array(image_size)[0]

        # Check the correctness of each object
        for i in range(len(text)):
            object = text[i]
            correct = 0 # flag to check if at least one instance of object correctly placed
            for bbox, score, label in zip(coords, scores, labels):
                if label == object:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    height, width =  abs(y2-y1), abs(x2-x1)

                    # Calculate centroid coordinates
                    centroid = (x1 + (width / 2), y1 + (height / 2))

                    # Check if the centroid lies in correct part of the image based on exp.
                    if experiment == "left_right_exp":
                        if i == 0:
                            if centroid[0] <= image_size[0]/2:
                                correct = 1
                                break
                        else:
                            if centroid[0] > image_size[0]/2:
                                correct = 1
                                break
                    elif experiment == "top_bottom_exp":
                        if i == 0:
                            if centroid[1] <= image_size[1]/2:
                                correct = 1
                                break
                        else:
                            if centroid[1] > image_size[1]/2:
                                correct = 1
                                break
            if correct != 1:
                # Missed this object in the image
                return 0
        # No objects missed
        return 1


def make_unique_filename() -> str:
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    rand_name = randomname.get_name()
    return f"{timestamp}_{rand_name}"


def step_00_build_img_cache(
    experiment: str = "xattn01",  # "baseline_exp", "xattn01", "xattn02", "xattn03"
    num_prompts: int = 1000,
    images_per_prompt: int = 4,
) -> None:
    prompts = set()
    while len(prompts) < num_prompts:
        prompts.add(get_random_prompt())
    base_dir = pathlib.Path("outputs", "cached_images", experiment,
                            make_unique_filename())
    print("Saving images to", base_dir, "...")
    model_kwargs = script.load_models_as_dict()

    with tqdm.tqdm(prompts, desc=experiment) as t:
        for vtup in t:
            img_dir = base_dir / vtup.prompt.replace(" ", "_") 
            img_dir.mkdir(parents=True, exist_ok=True)
            info_json = vtup._asdict()
            # Show the prompt in tqdm iterator
            t.set_postfix(prompt=vtup.prompt)

            # Dump info.json into the directory
            with open(img_dir / "info.json", "w") as f:
                json.dump(info_json, f, indent=4)

            if experiment == "baseline_exp":
                imgs = script.baseline_exp(vtup.prompt, **model_kwargs,
                        batch_size=images_per_prompt)
            elif experiment.startswith("xattn"):
                exp_kwargs = dict(
                    # 01: Our standard method. Negative prompting and 0.2 divider between left and right.
                    xattn01=dict(divider_size=0.2, neg_prompting=True),
                    # 02: Same as xattn01, but with no divider.
                    xattn02=dict(divider_size=0.0, neg_prompting=True),
                    # 03: Same as xattn01, but with no negative prompting.
                    xattn03=dict(divider_size=0.2, neg_prompting=False),
                    # 04: no negative prompting, no divider
                    xattn04=dict(divider_size=0.0, neg_prompting=False),
                )[experiment]
                exp_fns = dict(
                    left=script.left_right_exp,
                    right=script.left_right_exp,
                    above=script.top_bottom_exp,
                    below=script.top_bottom_exp,
                )
                exp_fn = exp_fns[vtup.relationship]
                if vtup.relationship in ["left", "above"]:
                    o1, o2 = vtup.obj1, vtup.obj2
                elif vtup.relationship in ["right", "below"]:
                    o1, o2 = vtup.obj2, vtup.obj1
                else:
                    raise ValueError("Invalid relationship:", vtup.relationship)

                # NICETOHAVE: disable tqdm progress bar for this call, or otherwise
                #   reduce verbosity of nested tqdm
                imgs = exp_fn(o1, o2, **model_kwargs, batch_size=images_per_prompt, **exp_kwargs)
            else:
                raise ValueError("Invalid experiment type:", experiment)

            for i, img in enumerate(imgs):
                # save image as i.png
                img.save(img_dir / f"{i}.png")

    
@torch.no_grad()
def step_01_build_detection_cache(cached_img_dir: pathlib.Path, device="cuda"):
    """Takes cached_img_dir as generated by build_img_cache and adds a new JSON file
    "detections.json" (DetectionTuple) to each subdirectory.
    This file contains a dictionary representation of DetectionTuple which contains
    detection bounding boxes and labels for every image in the directory.
    """
    cached_img_dir = pathlib.Path(cached_img_dir)
    assert cached_img_dir.is_dir()
         
    # instantiate model and load images
    processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32", device=device)
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model.to(device)

    # for each directory in cached_img_dir
    #   load info.json
    #   load images
    for img_dir in tqdm.tqdm(sorted(cached_img_dir.glob("*/")),
                             desc="Processing image directories"):
        with open(img_dir / "info.json") as f:
            info = json.load(f)
        vtup = data_types.VisorTuple(**info)

        # Load images and image filenames
        img_names = []
        images = []
        for img_path in sorted(img_dir.glob("*.png")):
            img_names.append(img_path.name)
            images.append(Image.open(img_path))

        # Get bounding boxes etc.
        texts = [[vtup.obj1, vtup.obj2]] * len(images)
        inputs = processor(text=texts, images=images, return_tensors="pt") 
        inputs = inputs.to(device)
        outputs = model(**inputs)

        target_sizes = [torch.Tensor(image.size[::-1]) for image in images]

        # Post-processor is cpu only. :P
        outputs.logits = outputs.logits.cpu()
        outputs.pred_boxes = outputs.pred_boxes.cpu()
        # Note: Seems like post-process keeps only the best label for each bounding box.
        #       So if cat_confidence=0.2 and dog_confidence=0.15, then we will not detect
        #           a dog at this bounding box.
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        # Write bounding boxes etc. to the same directory as DetectionTuple
        assert len(img_names) == len(results)
        for result in results:
            for k, v in result.items():
                result[k] = v.tolist()

        detect_tup = data_types.DetectionTuple(img_names=img_names, results=results)
        with open(img_dir / "boxes.json", "w") as f:
            json.dump(detect_tup._asdict(), f)


def step_02_compute_metrics_single(cached_img_dir: pathlib.Path):
    """Using info.json and detection.json previous step,
       compute the metrics for each image prompt and write to metrics.json.
    """
    cached_img_dir = pathlib.Path(cached_img_dir)

    individual_metrics = []
    # Enter each directory in cached_img_dir
    #  Load info.json
    #  Load detection.json
    #  Compute metrics: OA, VISOR-1, VISOR-2, VISOR-3, VISOR-4
    #       TODO: Compute both uncond and conditional versions of the previous metrics
    for img_dir in tqdm.tqdm(sorted(cached_img_dir.glob("*/")),
                             desc="Processing image directories"):
        if not img_dir.is_dir():
            continue
        with open(img_dir / "info.json") as f:
            info = json.load(f)
        vtup = data_types.VisorTuple(**info)
        with open(img_dir / "boxes.json") as f:
            boxes = json.load(f)
        dtup = data_types.DetectionTuple(**boxes)

        metrics = dict()
        metrics["visor"] = dtup.calc_visors(vtup)
        metrics["object_accuracy"] = dtup.calc_oas(vtup)
        metrics["oa1"] = dtup.calc_oa1(vtup)
        metrics["oa2"] = dtup.calc_oa2(vtup)
        metrics["_comment"] = (
            "visor1234 scores can be found in .visor.aggregates. "
            "oa1 and oa2 denote whether obj1 or obj2 are detected in the image."
        )
        # Now save metrics as JSON to metrics.json
        with open(img_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        individual_metrics.append(metrics)

    # Idea: take mean over each aggregate metric
    combined_metrics = dict()
    for met in individual_metrics:
        for met_k, met_dict in met.items():
            if not isinstance(met_dict, dict):
                continue
            if met_k not in combined_metrics:
                combined_metrics[met_k] = []
            combined_metrics[met_k].append(met_dict["aggregates"])

    # Now take mean over each aggregate metric
    for k, v in combined_metrics.items():
        combined_metrics[k] = np.mean(v, axis=0).tolist()

    # Now save combined metrics as JSON to metrics.json in cached_img_dir
    with open(cached_img_dir / "combined_metrics.json", "w") as f:
        s = json.dumps(combined_metrics, indent=4)
        f.write(s)
    print(f"Aggregate metrics for {cached_img_dir}:\n{s}")


if __name__ == "__main__":
    # step_02_compute_metrics_single("/scratch/steven/xattn_control/outputs/cached_images/xattn01/current")
    # step_02_compute_metrics_single("/scratch/steven/xattn_control/outputs/cached_images/baseline_exp/current")
    # step_00_build_img_cache("xattn01", num_prompts=1000)
    step_00_build_img_cache("xattn02", num_prompts=1000)
    step_00_build_img_cache("xattn03", num_prompts=1000)
    step_00_build_img_cache("xattn04", num_prompts=1000)