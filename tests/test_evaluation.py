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
    res = ol_eval.summarize(redirect_stdout=True)

    temp.close()
    return res

class TestEvaluation(unittest.TestCase):

    def testPerfectPrediction(self):

        def createPredPerfect(pred):
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            return pred

        res = testTemplate("8357.json", createPredPerfect)
        self.assertTrue(res[0]["value"] == 1, "Using GT as prediction does not give AP=1")

    def testMissingOneTP_ex8357(self):

        def createPredMissingOneTP_ex8357(pred):
            pred["annotations"] = [p for idx, p in enumerate(pred["annotations"]) if idx != 1]
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            return pred

        res = testTemplate("8357.json", createPredMissingOneTP_ex8357)
        self.assertTrue(res[13]["value"] == 0.875, "7 TP, 1 FN, 0 FP does not give recall=0.875")

    def testMissingTwoTP_ex8357(self):

        def createPredMissingTwoTP_ex8357(pred):
            pred["annotations"] = [p for idx, p in enumerate(pred["annotations"]) if not idx in [1,5]]
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            return pred

        res = testTemplate("8357.json", createPredMissingTwoTP_ex8357)
        self.assertTrue(res[13]["value"] == 0.750, "6 TP, 2 FN, 0 FP does not give recall=0.750")

    def testAddingOneFP_ex8357(self):

        def createPredAddingOneFP_ex8357(pred):
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
        self.assertTrue(abs(res[0]["value"] - 8./9) < 1e-12, f"8 TP,  0 FN, 1 FP does not give precision={8./9}")

    def testAddingOneFPtoExistingBBox_ex8357(self):

        def createPredAddingOneFPtoExistingBBox_ex8357(pred):
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            pred["annotations"][0]["description_ids"].append(2774)
            pred["annotations"][0]["scores"].append(2)
            return pred

        res = testTemplate("8357.json", createPredAddingOneFPtoExistingBBox_ex8357)
        self.assertTrue(abs(res[0]["value"] - 8./9) < 1e-12, f"8 TP,  0 FN, 1 FP does not give precision={8./9}")

    def testRemovingOneTPfromExistingBBox_ex8357(self):

        def createPredRemovingOneTPfromExistingBBox_ex8357(pred):
            for ann in pred["annotations"]:
                ann["scores"] = [1.0] * len(ann["description_ids"])
            pred["annotations"][2]["description_ids"] = pred["annotations"][2]["description_ids"][:1]
            pred["annotations"][2]["scores"] = pred["annotations"][2]["scores"][:1]
            return pred

        res = testTemplate("8357.json", createPredRemovingOneTPfromExistingBBox_ex8357)
        self.assertTrue(res[13]["value"] == 0.875, "7 TP, 1 FN, 0 FP does not give recall=0.875")


if __name__ == '__main__':
    unittest.main()
