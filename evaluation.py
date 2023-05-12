# Import libraries
import random
import os

from PIL import Image
import torch
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from script import *

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

relationships = ["to the left of", "above"]

# TODO: we need to create a list of objects that belong to different categories
#       desirably with completely different contexts

def generate_prompt(category_a, category_b, experiment="baseline_exp"):
    """
        takes two categories and returns a prompt that includes spatial relation between them
        the prompt is of the form "A/An <category_a> is <rel> a/an <category b>"
    """
    
    if experiment == "baseline_exp":
        relationship = random.choice(relationships)
    elif experiment == "left_right_exp":
        relationship = "to the left of"
    elif experiment == "top_bottom_exp":
        relationship = "above"

    article_a = "an" if category_a[0].lower() in "aeiou" else "a" # h
    article_b = "an" if category_b[0].lower() in "aeiou" else "a"

    prompt = f"{article_a} {category_a} is {relationship} {article_b} {category_b}"
    return prompt


def generate_prompts(categories, with_self=False, experiment="baseline_exp"):
    """
        generates prompts using a list of categories
        with_self: whether to use self-pairing
    """
    prompts = []
    for i in range(len(categories)):
        for j in range(i + with_self, len(categories)):
            category_a = categories[i]
            category_b = categories[j]
            prompt = generate_prompt(category_a, category_b, experiment=experiment)
            prompts.append((category_a, category_b, prompt))
    return prompts


def get_random_prompt(categories, experiment="baseline_exp"):
    """Get a random prompt. Self-pairing is not allowed."""
    category_a = random.choice(categories)
    category_b = random.choice(categories)
    while category_a == category_b:
        category_b = random.choice(categories)
    prompt = generate_prompt(category_a, category_b, experiment=experiment)
    return (category_a, category_b, prompt)


def dump_prompts_to_file(prompts, file_name):
    """
        dumps generated prompt to file
    """
    with open(file_name, "w") as file:
        for prompt in prompts:
            file.write(prompt + "\n")

# TODO: let SD generate images based on the category parings
# Generates SD image based on experiment type and prompt
unet, vae, clip, clip_tokenizer, device = load_models()
def generate_sd_images(prompt, prompt_num, images_per_prompt, experiment="baseline_exp"):
    if experiment == "baseline_exp":
        imgs = baseline_exp(prompt[2],
            unet=unet,
            vae=vae,
            device=device,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            seed=248396402679,
            batch_size=images_per_prompt,
        )
    elif experiment == "left_right_exp":
        imgs = left_right_exp(prompt[0], prompt[1],
            unet=unet,
            vae=vae,
            device=device,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            seed=248396402679,
            batch_size=images_per_prompt,
        )
    elif experiment == "top_bottom_exp":
        imgs = top_bottom_exp(prompt[0], prompt[1],
            unet=unet,
            vae=vae,
            device=device,
            clip=clip,
            clip_tokenizer=clip_tokenizer,
            seed=248396402679,
            batch_size=images_per_prompt,
        )
    
    if not os.path.exists(f'outputs/{experiment}'):
        os.makedirs(f'outputs/{experiment}')

    for i, img in enumerate(imgs):
        img.save(f"outputs/{experiment}/{str(prompt_num*images_per_prompt + i)}.png")

    return

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

if __name__ == "__main__":
    experiment = "left_right_exp" # "baseline_exp", "top_bottom_exp"
    num_images = 10 # Total number of images to test on
    images_per_prompt = 2 # Number of images to generate per prompt
    assert num_images%images_per_prompt == 0 # Number of images should be a multiple of images per prompt
    
    # We use OWL-ViT as the object detector
    # Tutorials here: https://huggingface.co/docs/transformers/model_doc/owlvit

    # instantiate model and load images
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Perform detection and scoring
    oa = []
    visor = []
    for prompt_num in range(int(num_images/images_per_prompt)):
        # Generate a random prompt
        prompt = get_random_prompt(categories=categories, experiment=experiment)
        print("Prompt: " + prompt[2])
        # Generate images_per_prompt images for prompt
        generate_sd_images(prompt, prompt_num, images_per_prompt, experiment=experiment)
        
        # For each image, do object detection and computer OA/VISOR score
        for image_num in range(prompt_num*images_per_prompt, prompt_num*images_per_prompt + images_per_prompt):
            image = Image.open(f"outputs/{experiment}/{image_num}.png")
            texts = [[prompt[0], prompt[1]]]
            inputs = processor(text=texts, images=image, return_tensors="pt")
            outputs = model(**inputs)

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([image.size[::-1]])

            # Convert outputs (bounding boxes and class logits) to COCO API
            results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

            i = 0  # Retrieve predictions for the first image for the corresponding text queries
            text = texts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            # Visualize bounding boxes
            score_threshold = 0.1
            box_coords = []
            confidence_scores = []
            confident_labels = []
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if score >= score_threshold:
                    # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                    box_coords.append(box)
                    confidence_scores.append(round(score.item(), 3))
                    confident_labels.append(text[label])

            visualize_bounding_boxes(coords=box_coords,
                                    scores=confidence_scores,
                                    labels=confident_labels,
                                    img=image,
                                    out_path=f"outputs/detection/{image_num}.png")

            # Compute OA and VISOR scores
            print()
            print("Image number: " + str(image_num))
            print("OA score: " + str(compute_OA_score(text, confident_labels)))
            oa.append(compute_OA_score(text, confident_labels))
            print("VISOR score: " + str(compute_VISOR_score(text, target_sizes, box_coords, confidence_scores, confident_labels, experiment=experiment)))
            visor.append(compute_VISOR_score(text, target_sizes, box_coords, confidence_scores, confident_labels, experiment=experiment))
            print()
    print("Overall OA score: " + str(round(np.mean(np.array(oa)), 2)))
    print("Overall VISOR score: " + str(round(np.mean(np.array(visor)), 2)))
