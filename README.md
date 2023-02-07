# OmniLabelTools (OLT) - Python tools for the OmniLabel benchmark

This repository provides a Python toolbox for working with the OmniLabel dataset/benchmark [(http://www.omnilabel.org)](http://www.omnilabel.org). The toolbox provides

* evaluation of prediction results
* visualization of ground truth and predictions
* extract basic statistics of the dataset annotations

[Install](#install) |
[Dataset setup](#dataset-setup) |
[Annotation format](#annotation-format) |
[Evaluate your results](#evaluate-your-results) |
[License](#license)

## Install

Install OmniLabelTools as:

``` bash
git clone https://www.github.com/samschulter/omnilabeltools
pip install omnilabeltools
```

You can also install in developer mode:

``` bash
pip install -e omnilabeltools
```

## Dataset setup

Please visit [http://www.omnilabel.org/download](http://www.omnilabel.org/download) for download and setup instructions. To verify the dataset setup, you can run the following two scripts to print some basic dataset statistics and visualize some examples:

``` python
olstats --path-to-json path/to/dataset/gt/json

olvis --path-to-json path/to/dataset/gt/json --path-to-imgs path/to/image/directories --path-output some/directory/to/store/visualizations
```

## Annotation format

In general, we try to follow the MS COCO dataset format as much as possible, with all annotations stored in one `json` file. Please see [http://www.omnilabel.org/task](http://www.omnilabel.org/task) for more details.

### Ground truth data

```json
{
    images: [
        {
            id              ... unique image ID
            file_name       ... path to image, relative to a given base directory (see above)
        },
        ...
    ],
    descriptions: [
        {
            id              ... unique description ID
            text            ... the text of the object description
            image_ids       ... list of image IDs for which this description is part of the label space
            anno_info       ... some metadata about the description
        },
        ...
    ],
    annotations: [        # Only for val sets. Not given in test set annotations!
        {
            id              ... unique annotation ID
            image_id        ... the image id this annotation belongs to
            bbox            ... the bounding box coordinates of the object (x,y,w,h)
            description_ids ... list of description IDs that refer to this object
	    },
        ...
    ]
}
```

### Submitting prediction results

**NB: The test server is not online at this time. Once online, prediction results are submitted in the following format:**

```json
[
    {
        image_id        ... the image id this predicted box belongs to
        bbox            ... the bounding box coordinates of the object (x,y,w,h)
        description_ids ... list of description IDs that refer to this object
        scores          ... list of confidences, one for each description
    },
    ...
]
```

## Evaluate your results

Here is some example code how to evaluate results:

``` python
from omnilabeltools import OmniLabel, OmniLabelEval

gt = OmniLabel(data_json_path)              # load ground truth dataset
dt = omniGt.load_res(res_json_path)         # load prediction results
ole = OmniLabelEval(gt, dt)
ole.params.resThrs = ...                    # set evaluation parameters as desired
ole.evaluate()
ole.accumulate()
ole.summarize()
```

We also provide a stand-alone script:

``` python
oleval --path-to-gt path/to/gt/json --path-to-res path/to/result/json
```

## License

This project is released under an [MIT License](LICENSE).
