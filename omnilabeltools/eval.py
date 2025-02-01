import numpy as np
import time
import datetime
import copy
from collections import defaultdict
from itertools import chain
from pycocotools.cocoeval import COCOeval, Params as ParamsCOCOAPI
from .omnilabel import OmniLabel


class OmniLabelEval(COCOeval):
    """
    Interface for evaluting predictions on the OmniLabel dataset. Use `OmniLabelEval` is as follows:

    gt = OmniLabel(data_json_path)             # load dataset with ground truth data
    dt = gt.load_res(res_json_path)            # load predictions (returns new `OmniLabel` instance)
    ole = OmniLabelEval(gt, dt)
    ole.params.recThrs = ...                   # set parameters as desired
    ole.evaluate()                             # run per image evaluation
    ole.accumulate()                           # accumulate per image results
    ole.summarize()                            # display summary metrics of results

    This code is developed based on [pycocotools](https://github.com/ppwwyyxx/cocoapi) and further
    adapted for the OmniLabel evaluation. The main changes are the introduction of object
    *descriptions* instead of object *categories*, along with the grouping of results based on type
    and length of such descriptions.

    The evaluation parameters are as follows (defaults in brackets, see `Params` for details):

    * imgIds        ... [all] N img ids to use for evaluation
    * iouThrs       ... [.5:.05:.95] T=10 IoU thresholds for evaluation
    * recThrs       ... [0:.01:1] R=101 recall thresholds for evaluation
    * areaRng       ... [...] A=4 object area ranges for evaluation
    * maxDets       ... [1 10 100] M=3 thresholds on max detections per image
    * descrLenghts  ... [see `Params`] D=6 groupings of object description lengths
    * descrTypes    ... [see `Params`] D=6 groupings of object description types
    * descrNames    ... [see `Params`] D=6 names of each object description group above

    Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.

    evaluate(): evaluates detections on every image and every description and concats the results
    into the `evalImgs` variable with fields:
    * dtIds         ... [1xD] id for each of the D detections (dt)
    * gtIds         ... [1xG] id for each of the G ground truths (gt)
    * dtMatches     ... [TxD] matching gt id at each IoU or 0
    * gtMatches     ... [TxG] matching dt id at each IoU or 0
    * dtScores      ... [1xD] confidence of each dt
    * gtIgnore      ... [1xG] ignore flag for each gt
    * dtIgnore      ... [TxD] ignore flag for each dt at each IoU

    accumulate(): accumulates the per-image evaluation results in `evalImgs` into the dictionary
    `eval` with fields:
    * params        ... parameters used for evaluation
    * date          ... date evaluation was performed
    * counts        ... [T,R,1,A,M,D] parameter dimensions (see above)
    * precision     ... [TxRx1xAxM,D] precision for every evaluation setting
    * recall        ... [Tx1xAxM,D] max recall for every evaluation setting

    Note: precision and recall==-1 for settings with no gt objects.
    """

    def __init__(self, gt: OmniLabel, dt: OmniLabel):
        """
        Initialize OmniLabelEval using OmniLabel instances for gt and dt

        Args:
            gt (OmniLabel): `OmniLabel` instance with ground truth annotations
            dt (OmniLabel): `OmniLabel` instance with detection results
        """
        self.omniGt = gt
        self.omniDt = dt

        self.evalImgs = defaultdict(list)     # evaluation results for every (img_id, descr_id)-pair
        self.eval = {}                        # accumulated evaluation results
        self._gts = defaultdict(list)         # gt for evaluation
        self._dts = defaultdict(list)         # dt for evaluation
        self.params = Params(iouType="bbox")  # parameters
        self._paramsEval = {}                 # parameters for evaluation
        self.stats = []                       # result summarization
        self.ious = {}                        # ious between all gts and dts

        self.params.imgIds = sorted(gt.image_ids)
        self.params.descrIds = sorted(gt.descr_ids)

    def _prepare(self):
        p = self.params

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation

        pImgIds_hash = {imgid: True for imgid in p.imgIds}  # Dict is faster to query with `in`

        for img_id, gt in self.omniGt.samples.items():
            if img_id not in pImgIds_hash:
                continue
            for instance in gt["instances"]:
                instance['ignore'] = 'iscrowd' in instance and instance['iscrowd']
                for di, descr_id in enumerate(instance["description_ids"]):
                    if di == 0:
                        instance_new = instance
                    else:
                        instance_new = {}  # Faster than copy.deepcopy()
                        instance_new.update(instance)
                    instance_new["descr_id"] = descr_id
                    self._gts[(img_id, descr_id)].append(instance_new)

        for img_id, dt in self.omniDt.samples.items():
            if img_id not in pImgIds_hash:
                continue
            for instance in dt["instances"]:
                assert len(instance["description_ids"]) == len(instance["scores"])
                for ii, (descr_id, score) in enumerate(zip(
                        instance["description_ids"], instance["scores"]
                )):
                    if ii == 0:
                        instance_new = instance
                    else:
                        instance_new = {}
                        instance_new.update(instance)
                    instance_new["descr_id"] = descr_id
                    instance_new["score"] = score
                    self._dts[(img_id, descr_id)].append(instance_new)

        self.evalImgs = defaultdict(list)   # evaluation results for every (img_id, descr_id)-pair
        self.eval = {}                      # accumulated evaluation results

        self.imgIdToLabelspace = {
            img_id: [d["id"] for d in gt["labelspace"]]
            for img_id, gt in self.omniGt.samples.items() if img_id in pImgIds_hash
        }

        self.descrIdToType = {}
        for img_id, gt in self.omniGt.samples.items():
            if img_id not in pImgIds_hash:
                continue
            pos_descr_ids = set(chain(*[box["description_ids"] for box in gt["instances"]]))
            pos_descr_ids_hash = {pid: True for pid in pos_descr_ids}
            for descr in gt["labelspace"]:
                type_str = descr["type"]
                if type_str == "D":
                    type_str += "p" if descr["id"] in pos_descr_ids_hash else "n"
                assert (descr["id"], img_id) not in self.descrIdToType
                self.descrIdToType[(descr["id"], img_id)] = type_str

        self.descrIdToDescrNumWords = {
            descr_id: len(self.omniGt.get_description(descr_id)["text"].split())
            for descr_id in self.omniGt.descr_ids
        }

        self.descrIdToNumRefBoxes = defaultdict(int)
        for img_id, gt in self.omniGt.samples.items():
            if img_id not in pImgIds_hash:
                continue
            for inst in gt["instances"]:
                for descr_id in inst["description_ids"]:
                    self.descrIdToNumRefBoxes[descr_id] += 1
        # NB: If at this point no box referred to a descrption, `self.descrIdToNumRefBoxes` will
        # return 0 for any description ID queried.

    def evaluate(self):
        """
        Run per-image evaluation on given images and store results (a list of dict) in
        `self.evalImgs`
        """
        tic = time.perf_counter()
        print("Running per image evaluation ... ", end="", flush=True)
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()

        # loop through images, area range, max detection number
        self.ious = {
            (imgId, descrId): self.computeIoU(imgId, descrId)
            for imgId in p.imgIds for descrId in self.imgIdToLabelspace[imgId]
        }
        maxDet = p.maxDets[-1]
        self.evalImgs = {
            (descrId, areaRng[0], areaRng[1], imgId):
            self.evaluateImg(imgId, descrId, areaRng, maxDet)
            for areaRng in p.areaRng
            for imgId in p.imgIds
            for descrId in self.imgIdToLabelspace[imgId]
        }
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.perf_counter()
        print("took {:0.2f} seconds".format(toc-tic))

    def evaluateImg(self, imgId, descrId, aRng, maxDet):
        """
        Perform evaluation for a single image and description

        Args:
            imgId (int)       ... image ID
            descrId (int)     ... description ID
            aRng (list(int))  ... area range (min, max) for bounding boxes
            maxDet (int)      ... maximum number of detections

        Returns:
            dict              ... single image results
        """
        # NB: We can re-use the base class' method, treating description ID as category ID
        ret = super().evaluateImg(imgId, descrId, aRng, maxDet)
        if ret is not None:
            ret["descr_id"] = ret["category_id"]
        return ret

    def accumulate(self, p=None):
        """
        Accumulate per-image evaluation results and store the result in `self.eval`

        Arg:
            p (Params): Input parameters for evaluation
        """
        print("Accumulating evaluation results ... ", end="", flush=True)
        tic = time.perf_counter()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        T = len(p.iouThrs)
        R = len(p.recThrs)
        A = len(p.areaRng)
        M = len(p.maxDets)
        D = len(p.descrGroups)
        precision = -np.ones((T, R, 1, A, M, D))  # -1 for the precision of absent description
        recall = -np.ones((T, 1, A, M, D))
        scores = -np.ones((T, R, 1, A, M, D))
        gtcount = -np.ones((A, M, D))

        # create dictionary for future indexing
        _pe = self._paramsEval
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [a for a in map(lambda x: tuple(x), p.areaRng) if a in setA]
        i_list = [i for i in p.imgIds if i in setI]
        # retrieve E at each description, area range, and max number of detections
        for a, a0 in enumerate(a_list):
            # Pre-filter the `self.evalImgs` dict, which makes the lookup faster below
            eval_imgs_a0 = {
                (key[0], key[3]): val for key, val in self.evalImgs.items()
                if key[1] == a0[0] and key[2] == a0[1]
            }

            for m, maxDet in enumerate(m_list):
                for di, descr_group in enumerate(p.descrGroups):
                    assert len(descr_group["len"]) == 2
                    E = [
                        eval_imgs_a0[(descrid, i)] if (descrid, i) in eval_imgs_a0 else None
                        for i in i_list for descrid in self.imgIdToLabelspace[i]
                        if (self.descrIdToType[(descrid, i)] in descr_group["type"])
                        and (self.descrIdToDescrNumWords[descrid] >= descr_group["len"][0])
                        and (self.descrIdToDescrNumWords[descrid] <= descr_group["len"][1])
                        and (self.descrIdToNumRefBoxes[descrid] >= descr_group["boxes"][0])
                        and (self.descrIdToNumRefBoxes[descrid] <= descr_group["boxes"][1])
                    ]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    gtcount[a, m, di] = npig
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm,  np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, 0, a, m, di] = rc[-1]
                        else:
                            recall[t, 0, a, m, di] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except Exception:
                            pass
                        precision[t, :, 0, a, m, di] = np.array(q)
                        scores[t, :, 0, a, m, di] = np.array(ss)
        self.eval = {
            "params": p,
            "counts": [T, R, A, M, D],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall":   recall,
            "scores": scores,
            "gtcount": gtcount,
        }
        toc = time.perf_counter()
        print("took {:0.2f} seconds".format(toc-tic))

    def summarize(self, verbose=True):
        """
        Summarize evaluation results in various metrics and groups (defined in `Params`)

        Arg:
            verbose (bool): If False, does not print the evaluation metrics and results

        Returns:
            list(dict): List of evaluation metric descriptions and result values
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", descr="all", maxDets=100):
            p = self.params
            iouStr = "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else "{:0.2f}".format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            dind = [g["name"] for g in p.descrGroups].index(descr)
            num_gt = int(self.eval["gtcount"][aind, mind, dind][0])
            if ap == 1:
                # dimension of precision: [TxRxKxAxMxD]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind, dind]
            else:
                # dimension of recall: [TxKxAxMxD]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind, dind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            metric_dict = {
                "metric": "AP" if ap == 1 else "AR",
                "iou": iouStr,
                "area": areaRng,
                "description": descr,
                "max_dets": maxDets,
            }
            return mean_s, metric_dict, num_gt

        def _summarize_all_metrics():

            metric_names = [grp["name"] for grp in self.params.descrGroups]
            compute_hm = "descr" in metric_names and "categ" in metric_names
            assert compute_hm, "Your parameter settings most contain both `descr` and `categ` to compute the final metric hm(descr, categ)"  # noqa: E501

            ret = []
            for grp in self.params.descrGroups:
                ret.append(_summarize(1, descr=grp["name"]))

            # Append standard metrics from COCO for both categories and descriptions
            ret.append(_summarize(1, descr="descr", iouThr=.5))
            ret.append(_summarize(1, descr="descr", iouThr=.75))
            ret.append(_summarize(1, descr="categ", iouThr=.5))
            ret.append(_summarize(1, descr="categ", iouThr=.75))
            ret.append(_summarize(0, descr="descr"))
            ret.append(_summarize(0, descr="categ"))

            # Separate tuples returned by `_summarize` into stats, metrics and num_gts. Add one
            # element that comes first for the final metric, see below
            stats = np.zeros((len(ret) + 1,))
            metrics = [None] * stats.size
            num_gts = [None] * stats.size
            for ii, (stat, metric, num_gt) in enumerate(ret):
                stats[ii + 1] = stat
                metrics[ii + 1] = metric
                num_gts[ii + 1] = num_gt

            # Overall metric: Harmonic mean of AP of descriptions and categories
            if num_gts[1] > 0 and num_gts[2] > 0:
                hm = (2 * stats[1] * stats[2]) / (stats[1] + stats[2] + 1e-5)
            else:
                hm = -1
            stats[0] = hm
            metrics[0] = copy.deepcopy(metrics[1])
            metrics[0]["description"] = "hm(descr,categ)"
            num_gts[0] = num_gts[1] + num_gts[2]

            return stats, metrics, num_gts

        def _print_metrics(results):
            res_str = (
                " {metric:s} @[ IoU={iou:<9} | area={area:>6s} | descr={descr:>15s} | "
                "maxDets={md:>3d} ][ num-gt={numgt:>5d} ] = {val:0.5f}"
            )
            for res in results:
                met = res["metric"]
                print(res_str.format(
                    metric=met["metric"],
                    iou=met["iou"],
                    area=met["area"],
                    descr=met["description"],
                    md=met["max_dets"],
                    numgt=res["numgt"],
                    val=res["value"]
                ))

        self.stats, self.metrics, num_gts = _summarize_all_metrics()
        ret = [
            {"metric": m, "value": v, "numgt": n}
            for m, v, n in zip(self.metrics, self.stats, num_gts)
        ]

        if verbose:
            _print_metrics(ret)

        return ret


class Params(ParamsCOCOAPI):
    """
    Collection of parameters that define the evaluation process. This is derived from the COCOAPI,
    see [here](https://github.com/ppwwyyxx/cocoapi/blob/71e284ef862300e4319aacd523a64c7f24750178/PythonAPI/pycocotools/cocoeval.py#L498.

    New parameters for grouping the results for object descriptions are defined:

    * 'categ'             ... Only plain object *categories*, as in other (open-vocabulary) detection datasets
    * 'descr'             ... Only free-form object *descriptions* (newly collected in the OmniLabel benchmark)
    * 'descr-pos'         ... Only free-form object *descriptions* that refer to objects in the image (excluding negative object descriptions)
    * 'descr-s'           ... Same as 'descr', but consider only descriptions up to 3 words (short)
    * 'descr-m'           ... Same as 'descr', but consider only descriptions from 4 to 8 words (medium)
    * 'descr-l'           ... Same as 'descr', but consider only descriptions longer than 8 words (long)
    """  # noqa: E501

    def __init__(self, iouType="segm"):
        super().__init__(iouType)
        max_len = 1e5
        self.descrGroups = [
            {"name": "categ",       "type":             ("C"), "len": [0, max_len], "boxes": [0, max_len]},  # noqa: E501
            {"name": "descr",       "type":      ("Dp", "Dn"), "len": [0, max_len], "boxes": [0, max_len]},  # noqa: E501
            {"name": "descr-pos",   "type":            ("Dp"), "len": [0, max_len], "boxes": [0, max_len]},  # noqa: E501
            # {"name": "descr-pos-1", "type":            ("Dp"), "len": [0, max_len], "boxes": [1, 1]},  # noqa: E501
            # {"name": "descr-pos-m", "type":            ("Dp"), "len": [0, max_len], "boxes": [2, max_len]},  # noqa: E501
            {"name": "descr-s",     "type":      ("Dp", "Dn"), "len": [0, 3],       "boxes": [0, max_len]},  # noqa: E501
            {"name": "descr-m",     "type":      ("Dp", "Dn"), "len": [4, 8],       "boxes": [0, max_len]},  # noqa: E501
            {"name": "descr-l",     "type":      ("Dp", "Dn"), "len": [9, max_len], "boxes": [0, max_len]},  # noqa: E501
        ]


def main_cli():
    import argparse
    from pathlib import Path

    _DESCR = "Evaluation tool for the OmniLabel benchmark (https://www.omnilabel.org)"
    _EPILO = """
For the ground truth JSON file (path-to-gt), use the files provided
by the official OmniLabel benchmark.

The JSON file containing prediction results, should also follow the
official data format, but only contain the `annotation` list. That is
the list of predicted bounding boxes, each with a list of scores for
each of the corresponding object predictions. For more details on the
data format, please visit: https://www.omnilabel.org/download
"""
    parser = argparse.ArgumentParser(
        description=_DESCR, epilog=_EPILO, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path-to-gt",
        required=True,
        metavar="PATH",
        help="Path to the OmniLabel ground truth JSON file"
    )
    parser.add_argument(
        "--path-to-res",
        required=True,
        metavar="PATH",
        help="Path to JSON file containing prediction results"
    )
    args = parser.parse_args()

    path_gt = Path(args.path_to_gt)
    assert path_gt.exists(), path_gt
    path_res = Path(args.path_to_res)
    assert path_res.exists(), path_res

    t_start = time.perf_counter()
    print("Loading ground truth & result files ... ", end="", flush=True)
    olgt = OmniLabel(path_json=path_gt)
    oldt = olgt.load_res(source=path_res)
    t_dur = time.perf_counter() - t_start
    print(f"took {t_dur:.2f} seconds")

    ol_eval = OmniLabelEval(gt=olgt, dt=oldt)
    ol_eval.evaluate()
    ol_eval.accumulate()
    ol_eval.summarize()
