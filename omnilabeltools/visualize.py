from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import platform
import os
import copy
from collections import defaultdict
from itertools import chain
from .omnilabel import OmniLabel


_RANDOM_COLORS = [
    (240, 153, 125, 255),
    (205, 233, 144, 255),
    (142, 167, 233, 255),
    (185, 243, 252, 255),
    (254, 222, 255, 255),
    (227, 172, 249, 255),
    (250, 218, 157, 255),
    (255, 212, 178, 255),
    (206, 237, 199, 255),
    (134, 200, 188, 255),
]


def _get_color(index: int):
    return _RANDOM_COLORS[index % len(_RANDOM_COLORS)]


def _get_font():
    if "OLTFONT" in os.environ:
        font_file = Path(os.environ["OLTFONT"]).absolute().resolve()
        assert font_file.exists(), \
            f"The font file given by environment variable 'OLTFONT' does not exist: '{font_file}'"
    elif platform.uname().system == "Linux":
        font_file = "FreeSans.ttf"
    elif platform.uname().system == "Darwin":
        font_file = "Arial Unicode.ttf"
    elif platform.uname().system == "Windows":
        font_file = "Arial.ttf"
    else:
        raise RuntimeError(
            f"Could not identify OS ('{platform.uname().system}') to find a truetype font file. "
            "You can set the environment variable 'OLTFONT' to a path to a truetype font ('.ttf')."
        )
    return font_file


class Visualizer:

    def __init__(self, image: Image.Image):
        assert isinstance(image, Image.Image), \
            f"Given image should be of type PIL.Image.Image but was '{type(image)}'"
        self.im = image
        self.draw = ImageDraw.Draw(image, mode="RGBA")
        self.font = ImageFont.truetype(_get_font(), 15)
        self.font_small = ImageFont.truetype(_get_font(), 10)

    def draw_instances(self, instances, labelspace, highlight_description_ids=None):
        id_to_text = {od["id"]: od["text"] for od in labelspace}
        for ii, instance in enumerate(instances):
            obj_x0, obj_y0, obj_wi, obj_hi = instance["bbox"]
            color = _get_color(ii)
            font = self.font
            line_width = 3
            text_fg_color = color
            text_box_fill_color = (47, 47, 47, 190)
            text_box_border_color = color

            if highlight_description_ids is not None and \
               all(descr_id not in highlight_description_ids
                   for descr_id in instance["description_ids"]):
                color = (47, 47, 47, 255)
                font = self.font_small
                line_width = 1
                text_fg_color = (47, 47, 47, 255)
                text_box_fill_color = None
                text_box_border_color = None

            # rectangle around object instance
            obj_x1 = obj_x0 + obj_wi
            obj_y1 = obj_y0 + obj_hi
            self.draw.rectangle(
                xy=[obj_x0, obj_y0, obj_x1, obj_y1], outline=color, width=line_width
            )

            # object descriptions
            if "scores" in instance:
                assert len(instance["scores"]) == len(instance["description_ids"])
                obj_descrs = [
                    f"[{score:.3f}] {id_to_text[descr_id]}"
                    for score, descr_id in zip(instance["scores"], instance["description_ids"])
                ]
            else:
                obj_descrs = [id_to_text[descr_id] for descr_id in instance["description_ids"]]
            # - longest description at the top (if no score is given)
            if "scores" not in instance:
                obj_descrs = sorted(obj_descrs, key=lambda txt: -len(txt))
            # - make long descriptions multiline
            txt_wi = self.draw.multiline_textbbox(xy=(0, 0), text=obj_descrs[0], font=font)[2]
            txt_box_wi = obj_wi * 1.5  # We let the text be twice as long as the box width
            txt_box_nc = int(len(obj_descrs[0]) * min(txt_box_wi / txt_wi, 1.0))

            def _split_into_lines(text, num_char_per_line):
                words = text.split()
                lines = []
                line = ""
                for ii, word in enumerate(words):
                    line += word if len(line) == 0 else " " + word
                    if len(line) > num_char_per_line:
                        lines.append(line)
                        line = ""
                if len(line) > 0:
                    lines.append(line)
                return lines

            obj_descrs = [_split_into_lines(descr, txt_box_nc) for descr in obj_descrs]
            # - list descriptions as bullet points
            obj_descrs = [
                ["+ " + line if li == 0 else "  " + line for li, line in enumerate(descr)]
                for descr in obj_descrs
            ]

            obj_descrs = "\n".join(chain(*obj_descrs))

            # Add offset between rectangle and text
            txt_x0 = obj_x0 + 9
            txt_y0 = obj_y0 + 7
            # Compute textbox around text
            text_bbox = list(self.draw.multiline_textbbox(
                xy=(txt_x0, txt_y0), text=obj_descrs, font=font
            ))
            text_bbox = [t + o for t, o in zip(text_bbox, [-7, -5, 5, 7])]
            # Move text and textbox if it extends beyond image boundary
            mv_left = max(0, text_bbox[2] - self.im.width)
            mv_up = max(0, text_bbox[3] - self.im.height)
            offsets = [-mv_left, -mv_up, -mv_left, -mv_up]
            text_bbox = [t + o for t, o in zip(text_bbox, offsets)]
            # NB: Offsets above are tuned for visualization on Linux

            if text_box_fill_color is not None:
                self.draw.rectangle(
                    xy=text_bbox,
                    outline=text_box_border_color,
                    fill=text_box_fill_color,
                    width=1
                )
            self.draw.multiline_text(
                xy=[txt_x0 - mv_left, txt_y0 - mv_up],
                text=obj_descrs,
                font=font,
                fill=text_fg_color
            )
            # TODO(sam): Improvement - for better visibility, we might want to move the text based
            # on the location and size of the bounding box.

    def get_image(self):
        return self.im


def visualize_image_sample(
        sample: Dict,
        path_imgs: str,
        highlight_description_ids: Optional[List[int]] = None,
        show_only_free_form_descriptions: bool = False,
        show_negative_free_form_descriptions: bool = False
):
    """
    Visualize a sample of the dataset, that is, an image along with the ground truth bounding boxes
    or the predicted bounding boxes. If visualizing predictions, the confidence scores are included.

    Args:
        sample (dict): A sample of the dataset, see `OmniLabel`
        path_imgs (str): A path to the base image directory
        highlight_description_ids (list(int)): List of description IDs to highlight
        show_only_free_form_descriptions (bool): Hide standard object categories in visualization
        show_negative_free_form_descriptions (bool): Show free-form object descriptions of the
            image's labelspace that do not refer to any object. Added below the image

    Returns:
        PIL.Image.Image
    """
    assert isinstance(sample, dict), \
        f"Given 'sample' must be of type 'dict', but found '{type(sample)}'"

    path_im = Path(path_imgs) / sample["file_name"]
    assert path_im.exists(), f"Image not found: '{path_im}'"
    vis = Visualizer(Image.open(path_im).convert("RGB"))

    instances = sample["instances"]
    if show_only_free_form_descriptions:
        descr_ids = [descr["id"] for descr in sample["labelspace"] if descr["type"] == "D"]
        instances = copy.deepcopy([  # deepcopy b/c the field 'description_ids' may change below
            inst for inst in instances
            if any(descr_id in descr_ids for descr_id in inst["description_ids"])
        ])
        for inst in instances:
            inst["description_ids"] = [
                descr_id for descr_id in inst["description_ids"] if descr_id in descr_ids
            ]

    vis.draw_instances(
        instances,
        sample["labelspace"],
        highlight_description_ids=highlight_description_ids,
    )

    vis_pil = vis.get_image()
    if show_negative_free_form_descriptions:
        # Get the text for negative descriptions
        descr_ids_pos = set(chain(*[inst["description_ids"] for inst in sample["instances"]]))
        neg_descrs = [
            descr for descr in sample["labelspace"]
            if descr["type"] == "D" and descr["id"] not in descr_ids_pos
        ]
        if len(neg_descrs) == 0:
            neg_text = "No negative free-form text descriptions for this image"
        else:
            neg_text = f"{len(neg_descrs)} negative free-form text descriptions:\n" + \
                "\n".join(["- \"" + descr["text"] + "\"" for descr in neg_descrs])

        # Draw the text below the actual image
        draw = ImageDraw.Draw(vis_pil, mode="RGB")
        text_w, text_h = draw.multiline_textsize(text=neg_text, font=vis.font)
        text_w += 10  # Left
        text_h += 15  # Top (5) & bottom (10)
        im_h = vis_pil.height + text_h
        im_w = max(vis_pil.width, text_w)
        vis_pil_pad = Image.new(vis_pil.mode, size=(im_w, im_h), color=(255, 255, 255))
        vis_pil_pad.paste(vis_pil, box=(0, 0))
        draw = ImageDraw.Draw(vis_pil_pad)
        draw.multiline_text(
            xy=(10, vis_pil.height + 5), text=neg_text, fill=(0, 0, 0), font=vis.font
        )
        vis_pil = vis_pil_pad

    return vis_pil


def gather_image_inds_from_description_inds(
        omnilabel: OmniLabel,
        description_inds: List[int],
        verbose: bool = True
):
    """
    Gather images that contain the given descriptions, if description refers to any object

    Args:
        omnilabel (OmniLabel): OmniLabel object holding the data
        description_inds (list[int]): List of description IDs
        verbose (bool): If True, prints when a description has no 'active' objects

    Returns:
        (list[int], list[list[int]]): First, a list of image IDs. Second, a list with, for each
            image ID, another list of description IDs that pointed to the corresponding image.
    """
    img_id_to_highlight_descr = defaultdict(list)
    for descr_id in description_inds:
        descr = omnilabel.get_description(descr_id)
        any_active = False
        for img_id in descr["image_ids"]:
            active_descr_ids = set(chain(*[
                inst["description_ids"] for inst in omnilabel.get_image_sample(img_id)["instances"]
            ]))
            if descr["id"] in active_descr_ids:
                any_active = True
                img_id_to_highlight_descr[img_id].append(descr["id"])
        if not any_active and verbose:
            print(f"Description '{descr_id}' does not refer to any object ... skipping")
    img_inds = list(img_id_to_highlight_descr.keys())
    highlight_descr_ids = [img_id_to_highlight_descr[img_id] for img_id in img_inds]
    return img_inds, highlight_descr_ids


def main_cli():
    import argparse
    from pathlib import Path
    import random

    _DESCR = ("Visualization tool for the OmniLabel benchmark dataset")
    parser = argparse.ArgumentParser(description=_DESCR)
    parser.add_argument(
        "--path-to-json", required=True, help="Path to the OmniLabel annotation JSON file"
    )
    parser.add_argument("--path-to-imgs", required=True, help="Directory pointing to images")
    parser.add_argument("--path-output", required=True, help="Directory to store visualization")
    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument(
        "--num", type=int, metavar="INT", default=10, help="Number of images to visualize"
    )
    parser_group.add_argument(
        "--image-index", type=int, nargs="+", metavar="IDX", help="Visualize specific image indices"
    )
    parser_group.add_argument(
        "--description-index", type=int, nargs="+", metavar="IDX",
        help="Visualize specific description indices"
    )
    parser.add_argument("--show-only-free-form-descriptions", action="store_true")
    parser.add_argument(
        "--show-negative-free-form-descriptions",
        action="store_true",
        help=("Show free-form object descriptions of the image's labelspace that do not refer to "
              "any object")
    )
    parser.add_argument(
        "--output-format", default="png", choices=["jpg", "png"], help="Default is 'png'"
    )
    args = parser.parse_args()

    ol = OmniLabel(args.path_to_json)
    assert ol.has_annotations, "This annotation JSON file does not contain annotations!"
    path_output = Path(args.path_output)
    path_output.mkdir(parents=True, exist_ok=True)

    highlight_descr_ids = None
    if args.image_index is not None:
        img_inds = args.image_index
    elif args.description_index is not None:
        img_inds, highlight_descr_ids = \
            gather_image_inds_from_description_inds(ol, args.description_index)
    else:
        img_inds = random.sample(ol.image_ids, args.num)

    for ii, idx in enumerate(img_inds):
        sample = ol.get_image_sample(idx)
        im = visualize_image_sample(
            sample=sample,
            path_imgs=args.path_to_imgs,
            highlight_description_ids=highlight_descr_ids,
            show_only_free_form_descriptions=args.show_only_free_form_descriptions,
            show_negative_free_form_descriptions=args.show_negative_free_form_descriptions
        )
        fn_out = path_output / "{imgid:010d}.{fmt:s}".format(
            imgid=sample["id"], fmt=args.output_format)
        im.save(fn_out)
        print(f"[{ii+1}/{len(img_inds)}] Stored visualization at '{fn_out}'")
