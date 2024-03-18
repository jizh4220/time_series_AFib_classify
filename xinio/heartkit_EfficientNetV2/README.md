
**Reference**: <a href="https://github.com/AmbiqAI/heartkit" target="_blank">https://github.com/AmbiqAI/heartkit</a>

**Data**: <a href="https://physionet.org/content/icentia11k-continuous-ecg/1.0/" target="_blank">https://physionet.org/content/icentia11k-continuous-ecg/1.0/</a>

---

## <span class="sk-h2-span">Requirement

* [Python ^3.11+](https://www.python.org)
* [Poetry ^1.6.1+](https://python-poetry.org/docs/#installation)

## <span class="sk-h2-span">Installation</span>

To set up the `DigitalHealth` environment and install the necessary packages, you can follow these steps in your terminal:

1. Create a new conda environment named `DigitalHealth` with Python 3.11:

    ```bash
    conda create -n DigitalHealth -c conda-forge python=3.11
    ```

2. Activate the `DigitalHealth` environment:

    ```bash
    conda activate DigitalHealth
    ```

3. Clone the `heartkit` repository from GitHub:

    ```bash
    git clone https://github.com/AmbiqAI/heartkit.git
    ```

4. Navigate to the `heartkit` directory:

    ```bash
    cd heartkit/
    ```

5. Install `poetry`, a tool for Python project and dependency management:

    ```bash
    pip install poetry
    ```

6. Install the dependencies using `poetry`:

    ```bash
    poetry install
    ```

After running these commands, your `DigitalHealth` environment should be set up with the necessary packages and the `heartkit` project.

## <span class="sk-h2-span">Download Incentia11k</span>

```python
from pathlib import Path
import heartkit as hk
import tensorflow as tf


dataset = hk.datasets.IcentiaDataset(
    ds_path=Path("~/heartkit/datasets/"), # Specify the path to the dataset
    task="classification", # Specify the task to perform
    frame_size=256, # Specify the frame size for the data
    target_rate=300, # Specify the target sampling rate for the data
    spec=(tf.TensorSpec(shape=(None, 256, 1), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32)) # Specify the input and output specifications for the data
) # Create an instance of the class with the required parameters
dataset.download(
    num_workers=4,
    force=True
)
```

## Incentia11k Data Structure

You can find the h5 files for the Incentia11k dataset in the following directory: `~/heartkit/datasets/icentia11k`.

The structure of the patient data is as follows:

* `./p00000.h5`: This file stores all segments for the first patient. There are at most 50 segments.
* `./p10999.h5`: This file stores all segments for the last patient.
* `./p10000.h5` to `./p10999.h5`: These files are reserved for separate testing.

Please note that each `.h5` file corresponds to a specific patient's data.

To access every segment under a specific patient, we use p10000 as an example:

```python
import os
import tensorflow as tf

from heartkit.tasks import TaskFactory
from typing import Type, TypeVar
from argdantic import ArgField, ArgParser
from pydantic import BaseModel
from heartkit.utils import env_flag, set_random_seed, setup_logger

from heartkit.tasks.AFIB_Ident.utils import (
    create_model,
    load_datasets,
    load_test_datasets,
    load_train_datasets,
    prepare,
)

from heartkit.tasks.AFIB_Ident.utils import (
    create_model,
    load_datasets,
    load_test_datasets,
    load_train_datasets,
    prepare,
)

from heartkit.defines import (
    HKDemoParams,
    HKDownloadParams,
    HKExportParams,
    HKMode,
    HKTestParams,
    HKTrainParams,
)


from heartkit.tasks.AFIB_Ident.defines import (
    get_class_mapping,
    get_class_names,
    get_class_shape,
    get_classes,
    get_feat_shape,
)

cli = ArgParser()
B = TypeVar("B", bound=BaseModel)


def parse_content(cls: Type[B], content: str) -> B:
    """Parse file or raw content into Pydantic model.

    Args:
        cls (B): Pydantic model subclasss
        content (str): File path or raw content

    Returns:
        B: Pydantic model subclass instance
    """
    if os.path.isfile(content):
        with open(content, "r", encoding="utf-8") as f:
            content = f.read()

    return cls.model_validate_json(json_data=content)

config = 'configs/arrhythmia-class-2.json'
params = parse_content(HKDemoParams, config)

class_names = get_class_names(params.num_classes)
class_map = get_class_mapping(params.num_classes)
input_spec = (
    tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
    tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
)

ds = load_datasets(
    ds_path=params.ds_path,
    frame_size=params.frame_size,
    sampling_rate=params.sampling_rate,
    class_map=class_map,
    spec=input_spec,
    datasets=params.datasets,
)[0]

patient_ids = ds.get_test_patient_ids()
single_pat_gen = ds.uniform_patient_generator(patient_ids=patient_ids[patient_ids=="10000"], repeat=False, shuffle=False)
for _, segments in single_pat_gen:
  segment_id = np.random.choice(list(segments.keys())) # randomly pick a segment
  segment = segments[segment_id]
  # get the overall size of current segment, _sID
  segment_size = segment["data"].shape[0]
  # get all the peak rlabels for current segment
  rlabels = segment["rlabels"][:]
```

## EfficientNetV2 for 2-Class Classification

### Model Architecture

The arrhythmia models utilizes a variation of [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) that is adapted for 1-D time series data. The model is a 1-D CNN built using [MBConv](https://paperswithcode.com/method/inverted-residual-block) style blocks that incorporate expansion, inverted residuals, and squeeze and excitation layers. Furthermore, longer filter and stride lengths are utilized in the initial layers to capture more temporal dependencies.

### Preprocessing

The models are trained directly on single channel ECG data. No feature extraction is performed other than applying a bandpass filter ([1, 30]) to remove noise followed by downsampling. The signal is then z-normed. We also add a small epsilon value to the standard deviation to avoid division by zero.


    | CLASS    | LABELS           |
    | -------- | ---------------- |
    | 0        | NSR              |
    | 1        | AFIB, AFL        |

### Evaluation Metrics

For each dataset, 10% of the data is held out for testing (p10000~p10999). From the remaining, 20% of the data is randomly selected for validation. There is no mixing of subjects between the training, validation, and test sets. Furthermore, the test set is held fixed while training and validation are randomly split during training. We evaluate the models performance using a variety of metrics including loss, accuracy, and F1 score.

### Metrics

    | Metric   | Baseline | 
    | -------- | -------- |
    | Accuracy | 94.38%   |
    | F1 Score | 94.37%   |
    | Precision | 94.43%  |
    | Sensitivity | 94.38%  |
    | AUC-ROC | 94.36%  |


## Incentia11k Model Validation
```python
config = 'configs/arrhythmia-class-2.json'
params = parse_content(HKTestParams, config) #load the test params

task="AFIB_Ident"
task_handler = TaskFactory.get(task)
task_handler.evaluate(params) # evaluate the model, 

```

## TODO (For Nicole)

use 