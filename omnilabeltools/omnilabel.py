from typing import Union, List
from pathlib import Path
import json
import copy
from collections import defaultdict


class OmniLabel:
    """
    Container to load an OmniLabel annotation file (`json`) and structure the data for easy use as
    data loader and evaluating prediction results.
    """

    def __init__(self, path_json: Union[str, Path]):
        """
        Arg:
            path_json (Path, str): Path to OmniLabel ground truth file (`json`)
        """
        if isinstance(path_json, str):
            path_json = Path(path_json)
        assert path_json.exists(), f"Path to JSON '{path_json}' does not work!"
        with open(path_json, "r") as fid:
            data_json = json.load(fid)
        assert "images" in data_json, "Faulty JSON file: 'images' key missing"
        assert "descriptions" in data_json, "Faulty JSON file: 'descriptions' key missing"
        self.has_boxes = has_boxes = "annotations" in data_json

        # Re-structure the data for easier access
        imgid_to_descrs = defaultdict(list)
        for descr in data_json["descriptions"]:
            descr["type"] = "D" if descr["anno_info"]["type"] == "object_description" else "C"
            for imgid in descr["image_ids"]:
                imgid_to_descrs[imgid].append(descr)
            # Remove redundant info to keep samples lean and clean
            descr.pop("anno_info")
        self.descr_id_to_descr = {descr["id"]: descr for descr in data_json["descriptions"]}

        if has_boxes:
            imgid_to_boxes = defaultdict(list)
            for box in data_json["annotations"]:
                box["area"] = max(0, box["bbox"][2] * box["bbox"][3])
                imgid_to_boxes[box["image_id"]].append(box)

        for img in data_json["images"]:
            img["labelspace"] = imgid_to_descrs[img["id"]]
            if has_boxes:
                img["instances"] = imgid_to_boxes[img["id"]]
        self.samples = {d["id"]: d for d in data_json["images"]}

    def load_res(self, source: Union[str, Path, List[dict]]):
        """
        Loads the given prediction results, instantiates a copy of the `OmniLabel` instance
        (of `self`) and replaces the ground truth boxes with the predictions.

        Args:
            source (Path, str, List[dict]): Either, the source a path to a json file containing the
                prediction results, or a list of the results themselves

        Returns:
            OmniLabel: A deepcopy of the instance (`self`) itself, with ground truth boxes replaced
                by predictions
        """
        assert isinstance(source, (str, Path, list))
        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            assert source.exists(), f"Path to JSON '{source}' does not work!"
            with open(source, "r") as fid:
                result_json = json.load(fid)
            assert isinstance(result_json, list), "Faulty JSON file: not a list"
        else:
            assert isinstance(source, list)
            result_json = source

        res = copy.deepcopy(self)
        imgid_to_boxes = defaultdict(list)
        cnt = 123  # This should not be 0 to avoid confusion in the evaluator
        for box in result_json:
            box["area"] = max(0, box["bbox"][2] * box["bbox"][3])
            box["id"] = cnt
            imgid_to_boxes[box["image_id"]].append(box)
            cnt += 1
        for img_id, img in res.samples.items():
            img["instances"] = imgid_to_boxes[img_id]
        return res

    @property
    def num_images(self):
        return len(self.samples)

    @property
    def image_ids(self):
        return sorted(list(self.samples.keys()))

    @property
    def descr_ids(self):
        return sorted(list(self.descr_id_to_descr.keys()))

    @property
    def has_annotations(self):
        return self.has_boxes

    def get_image_sample(self, image_id: int):
        """
        Retrieve image sample by image_id. The image sample is a dict with fields for `file_name`,
        `labelspace` and `instances` (only for validation set containing ground truth).

        Arg:
            image_id (int)

        Returns:
            dict
        """
        assert isinstance(image_id, int), f"image_id should by integer, but is '{type(image_id)}'"
        assert image_id in self.samples, f"image_id '{image_id}' is invalid, not found"
        return self.samples[image_id]

    def get_description(self, description_id: int):
        """
        Retrieve description by description_id.

        Arg:
            description_id (int)

        Returns:
            dict
        """
        assert isinstance(description_id, int), \
            f"description_id should by integer, but is '{type(description_id)}'"
        assert description_id in self.descr_id_to_descr, \
            f"description_id '{description_id}' is invalid, not found"
        return self.descr_id_to_descr[description_id]
