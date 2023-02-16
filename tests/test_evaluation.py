import unittest
from pathlib import Path
import tempfile
import json
from omnilabeltools import OmniLabel, OmniLabelEval

tests_dir = Path(__file__).parent


def testTemplate(filename, createPred):
    gt_json = tests_dir / filename
    olgt = OmniLabel(path_json=gt_json)

    with open(gt_json) as fid:
        pred = json.load(fid)
    pred = createPred(pred)

    temp = tempfile.NamedTemporaryFile()
    with open(temp.name, "w") as fid:
        json.dump(pred["annotations"], fid)
    oldt = olgt.load_res(temp.name)

    ol_eval = OmniLabelEval(gt=olgt, dt=oldt)
    ol_eval.evaluate()
    ol_eval.accumulate()
    res = ol_eval.summarize(verbose=False)

    temp.close()
    return res


class TestEvaluation(unittest.TestCase):

    def testPerfectPrediction(self):

        def createPredPerfect(pred):
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            return pred

        res = testTemplate("8357.json", createPredPerfect)
        self.assertAlmostEqual(
            res[0]["value"], 1.0, delta=1e-5, msg="Using GT as prediction does not give AP=1"
        )

    def testMissingOneTP_ex8357(self):

        def createPredMissingOneTP_ex8357(pred):
            # Removing one of six standard category object descriptions
            pred["annotations"] = [p for idx, p in enumerate(pred["annotations"]) if idx != 1]
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            return pred

        res = testTemplate("8357.json", createPredMissingOneTP_ex8357)
        self.assertAlmostEqual(
            res[12]["value"], 5/6, delta=1e-5, msg="5 TP, 1 FN, 0 FP does not give recall=0.8333"
        )

    def testMissingTwoTP_ex8357(self):

        def createPredMissingTwoTP_ex8357(pred):
            # Removing two of six standard category object descriptions
            pred["annotations"] = [
                p for idx, p in enumerate(pred["annotations"]) if idx not in [1, 5]
            ]
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            return pred

        res = testTemplate("8357.json", createPredMissingTwoTP_ex8357)
        self.assertAlmostEqual(
            res[12]["value"], 4/6, delta=1e-5, msg="4 TP, 2 FN, 0 FP does not give recall=0.66666"
        )

    def testAddingOneFP_ex8357(self):

        def createPredAddingOneFP_ex8357(pred):
            # Add one FP for standard category object descriptions
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            pred["annotations"].append({
                    "image_id": 8357,
                    "description_ids": [8580],
                    "bbox": [14, 18, 2, 68],
                    "scores": [2]
            })
            return pred

        res = testTemplate("8357.json", createPredAddingOneFP_ex8357)
        # 2 free-form descriptions, 6 standard descriptions, 1 false-positive standard description
        # Target: harmonic mean of 1.0 and 6/7
        target = (2*1.0*6/7) / (1.0 + 6/7)
        self.assertAlmostEqual(
            res[0]["value"],
            target,
            delta=1e-5,
            msg="Adding 1 FP standard-category prediction gives wrong final result (harmonic mean)"
        )

    def testAddingOneFPtoExistingBBox_ex8357(self):

        def createPredAddingOneFPtoExistingBBox_ex8357(pred):
            # Add one FP for free-form object descriptions
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            pred["annotations"][0]["description_ids"].append(2774)
            pred["annotations"][0]["scores"].append(2)
            return pred

        res = testTemplate("8357.json", createPredAddingOneFPtoExistingBBox_ex8357)
        # 2 free-form descriptions, 6 standard descriptions, 1 false-positive free-form description
        # Target: harmonic mean of 2/3 and 1.0
        target = (2*2/3*1) / (2/3 + 1)
        self.assertAlmostEqual(
            res[0]["value"],
            target,
            delta=1e-5,
            msg="Adding 1 FP free-form prediction gives wrong final result (harmonic mean)"
        )

    def testRemovingOneTPfromExistingBBox_ex8357(self):

        def createPredRemovingOneTPfromExistingBBox_ex8357(pred):
            # Removing one of two free-form object descriptions
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            pred["annotations"][2]["description_ids"] = \
                pred["annotations"][2]["description_ids"][:1]
            pred["annotations"][2]["scores"] = pred["annotations"][2]["scores"][:1]
            return pred

        res = testTemplate("8357.json", createPredRemovingOneTPfromExistingBBox_ex8357)
        self.assertAlmostEqual(
            res[11]["value"], 1/2, delta=1e-5, msg="1 TP, 1 FN, 0 FP does not give recall=0.5"
        )

    def testAPDescrPosWhenAddingOneFPOnNegDescr_ex8357(self):

        def createPredAddingOneFP_ex8357(pred):
            # Turn annotations into predictions by adding scores
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])

            # Add one FP for a negative free-form object description
            from itertools import chain
            pos_descr_ids = set(chain(*[box["description_ids"] for box in pred["annotations"]]))
            neg_freeform_descr_ids = [
                descr["id"] for descr in pred["descriptions"]
                if descr["anno_info"]["type"] == "object_description"
                and descr["id"] not in pos_descr_ids
            ]
            assert len(neg_freeform_descr_ids) > 0
            neg_freeform_descr_id = neg_freeform_descr_ids[0]
            pred["annotations"].append({
                    "image_id": 8357,
                    "description_ids": [neg_freeform_descr_id],
                    "bbox": [14, 18, 2, 68],
                    "scores": [2.0]  # Higher score than other detections!
            })
            return pred

        res = testTemplate("8357.json", createPredAddingOneFP_ex8357)

        # Adding one FP for a negative free-form description should not affect the `AP-descr-Pos`
        # metric, which ignores negative descritpions. But it should reduce the `AP-descr` metric.
        # In this case, we have 2 ground truth free-form descriptions, plus one FP > Target for
        # `AP-descr` is then 2/3 and target for `AP-descr-Pos` is 1.0
        with self.subTest():
            self.assertAlmostEqual(
                res[2]["value"], 2/3, delta=1e-5, msg="2 TP and 1 FP does not give AP=2/3"
            )
        with self.subTest():
            self.assertAlmostEqual(
                res[3]["value"],
                1.0,
                delta=1e-5,
                msg="Adding 1 FP of negative free-from description should not affect 'AP-descr-Pos'"
            )


if __name__ == '__main__':
    unittest.main()
