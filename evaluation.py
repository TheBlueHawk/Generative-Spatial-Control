import random

from PIL import Image
import torch
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from transformers import OwlViTProcessor, OwlViTForObjectDetection

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


relationships = ["to the left of", "to the right of", "above", "below"]


# TODO: we need to create a list of objects that belong to different categories
#       desirably with completely different contexts

# TODO: let SD generate images based on the category parings

def generate_prompt(category_a, category_b):
    """
        takes two categories and returns a prompt that includes spatial relation between them
        the prompt is of the form "A/An <category_a> is <rel> a/an <category b>"
    """
    relationship = random.choice(relationships)

    article_a = "an" if category_a[0].lower() in "aeiou" else "a" # h
    article_b = "an" if category_b[0].lower() in "aeiou" else "a"

    prompt = f"{article_a} {category_a} is {relationship} {article_b} {category_b}"
    return prompt


def generate_prompts(categories, with_self=False):
    """
        generates prompts using a list of categories
        with_self: whether to use self-pairing
    """
    prompts = []
    for i in range(len(categories)):
        for j in range(i + with_self, len(categories)):
            category_a = categories[i]
            category_b = categories[j]
            prompt = generate_prompt(category_a, category_b)
            prompts.append(prompt)
    return prompts


def dump_prompts_to_file(prompts, file_name):
    """
        dumps generated prompt to file
    """
    with open(file_name, "w") as file:
        for prompt in prompts:
            file.write(prompt + "\n")


def visualize_bounding_boxes(coords, scores, labels, img, save=False, out_name=None):
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
    if save:
        assert out_name is not None
        plt.savefig(f"detection/{out_name}.png")

    plt.show()


if __name__ == "__main__":
    # we use OWL-ViT as the object detector
    # tutorials here: https://huggingface.co/docs/transformers/model_doc/owlvit

    # instantiate model and load images
    path = Path("outputs/empty_string_middle_prompt")
    files = [f for f in path.glob("*.png")]

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # save a bunch of figures
    for l in range(5):
        image = Image.open(files[l])
        texts = [["cat", "dog"]] # TODO custom texts
        inputs = processor(text=texts, images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])

        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # visualize bounding boxes
        score_threshold = 0.1
        box_coords = []
        confidence_scores = []
        confident_labels = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                box_coords.append(box)
                confidence_scores.append(round(score.item(), 3))
                confident_labels.append(text[label])

        visualize_bounding_boxes(coords=box_coords,
                                 scores=confidence_scores,
                                 labels=confident_labels,
                                 img=image,
                                 save=True,
                                 out_name=f"sample_{l}")
