import argparse
import numpy as np
from itertools import chain
from collections import defaultdict
from .omnilabel import OmniLabel


def main_cli():
    _DESCR = ("Statistics about the OmniLabel dataset")
    parser = argparse.ArgumentParser(description=_DESCR)
    parser.add_argument(
        "--path-to-json",
        required=True,
        metavar="PATH",
        help="Path to a OmniLabel annotation JSON file"
    )
    args = parser.parse_args()

    ol = OmniLabel(args.path_to_json)
    print(f"Basic statistics about '{args.path_to_json}'")

    if not ol.has_annotations:
        print("\nNB: THIS DATASET JSON HAS NO ANNOTATIONS (E.G., TEST SET) > "
              "ONLY SHOWING LIMITED STATISTICS\n")

    print(f"Number of images: {ol.num_images}")
    img_ids = ol.image_ids

    # Boxes
    if ol.has_annotations:
        boxes_per_img = [ol.get_image_sample(idx)["instances"] for idx in img_ids]
        num_boxes = np.array([len(boxes) for boxes in boxes_per_img])
        print(f"Number of boxes: {num_boxes.sum()}")
        print(f"Number of boxes per image: {num_boxes.mean():.1f} +- {num_boxes.std():.1f} "
              f"(min={num_boxes.min()}, max={num_boxes.max()})")

    # Descriptions
    descr_ids = set(chain(*[
        [x["id"] for x in ol.get_image_sample(idx)["labelspace"]] for idx in img_ids
    ]))
    descrs = [ol.get_description(x) for x in descr_ids]
    descrs_C = [d for d in descrs if d["type"] == "C"]
    descrs_D = [d for d in descrs if d["type"] == "D"]
    print(f"Number of object descriptions: {len(descrs)} "
          f"(free-form={len(descrs_D)}, categories={len(descrs_C)})")

    # Length of object descriptions (only object descriptions - min, max, mean, std)
    descr_l_c = np.array([len(descr["text"]) for descr in descrs if descr["type"] == "D"])
    descr_l_w = np.array([len(descr["text"].split()) for descr in descrs if descr["type"] == "D"])
    print(f"Description lenghts (in 'chars'): {descr_l_c.mean():.1f} +- {descr_l_c.std():.1f} "
          f"(min={descr_l_c.min()}, max={descr_l_c.max()})")
    print(f"Description lenghts (in 'words'): {descr_l_w.mean():.1f} +- {descr_l_w.std():.1f} "
          f"(min={descr_l_w.min()}, max={descr_l_w.max()})")

    # Label spaces per image (includes 'negatives/zero-instance', descriptions)
    lss = [ol.get_image_sample(idx)["labelspace"] for idx in img_ids]
    ls_sizes = np.array([len(ls) for ls in lss])
    print(f"Labelspace sizes per image: {ls_sizes.mean():.1f} +- {ls_sizes.std():.1f} "
          f"(min={ls_sizes.min()}, max={ls_sizes.max()})")
    lss_C = [[descr for descr in ls if descr["type"] == "C"] for ls in lss]
    lss_D = [[descr for descr in ls if descr["type"] == "D"] for ls in lss]
    ls_sizes_C = np.array([len(ls) for ls in lss_C])
    print(f"Labelspace sizes (standard categories) per image: {ls_sizes_C.mean():.1f} "
          f"+- {ls_sizes_C.std():.1f} (min={ls_sizes_C.min()}, max={ls_sizes_C.max()})")
    ls_sizes_D = np.array([len(ls) for ls in lss_D])
    print(f"Labelspace sizes (object descriptions) per image: {ls_sizes_D.mean():.1f} "
          f"+- {ls_sizes_D.std():.1f} (min={ls_sizes_D.min()}, max={ls_sizes_D.max()})")

    # "Active" label spaces per image (only visible object instances)
    if ol.has_annotations:
        alss = [
            set(chain(*[inst["description_ids"] for inst in ol.get_image_sample(idx)["instances"]]))
            for idx in img_ids
        ]
        als_sizes = np.array([len(ls) for ls in alss])
        print(f"Active labelspace sizes per image: {als_sizes.mean():.1f} +- {als_sizes.std():.1f} "
              f"(min={als_sizes.min()}, max={als_sizes.max()})")
        descr_ids_C = {descr["id"]: None for descr in descrs_C}  # Faster for lookup than a list
        alss_C = [[descr_id for descr_id in als if descr_id in descr_ids_C] for als in alss]
        als_sizes_C = np.array([len(als) for als in alss_C])
        print(f"Active labelspace sizes (standard categories) per image: {als_sizes_C.mean():.1f} "
              f"+- {als_sizes_C.std():.1f} (min={als_sizes_C.min()}, max={als_sizes_C.max()})")
        descr_ids_D = {descr["id"]: None for descr in descrs_D}
        alss_D = [[descr_id for descr_id in als if descr_id in descr_ids_D] for als in alss]
        als_sizes_D = np.array([len(als) for als in alss_D])
        print(f"Active labelspace sizes (object descriptions) per image: {als_sizes_D.mean():.1f} "
              f"+- {als_sizes_D.std():.1f} (min={als_sizes_D.min()}, max={als_sizes_D.max()})")

    # Number of boxes per object description
    if ol.has_annotations:
        boxes = list(chain(*[ol.get_image_sample(idx)["instances"] for idx in img_ids]))
        box_ids = set([box["id"] for box in boxes])
        assert len(box_ids) == len(boxes)  # Sanity check
        descrid_to_boxids = defaultdict(list)
        for box in boxes:
            for descr_id in box["description_ids"]:
                descrid_to_boxids[descr_id].append(box["id"])
        nb_per_descr = np.array([len(boxids) for boxids in descrid_to_boxids.values()])
        print(f"Number of boxes per description: {nb_per_descr.mean():.1f} "
              f"+- {nb_per_descr.std():.1f} (min={nb_per_descr.min()}, max={nb_per_descr.max()})")
        nb_per_descr_C = np.array([
            len(boxids) for descrid, boxids in descrid_to_boxids.items() if descrid in descr_ids_C
        ])
        print(f"Number of boxes per description (standard categories): {nb_per_descr_C.mean():.1f} "
              f"+- {nb_per_descr_C.std():.1f} (min={nb_per_descr_C.min()}, "
              f"max={nb_per_descr_C.max()})")
        nb_per_descr_D = np.array([
            len(boxids) for descrid, boxids in descrid_to_boxids.items() if descrid in descr_ids_D
        ])
        print(f"Number of boxes per description (object descriptions): {nb_per_descr_D.mean():.1f} "
              f"+- {nb_per_descr_D.std():.1f} (min={nb_per_descr_D.min()}, "
              f"max={nb_per_descr_D.max()})")
