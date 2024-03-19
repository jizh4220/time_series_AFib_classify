## Basically we have a `arrhythmia` task hanlder `icentia11k` dataloader

- `arrhythmia`: train, evaluate, export, demo are four separate methods that require loading the datasets, loading the model
- `icentia11k`: rhythm_data_generator() is the deep core method that samples signal data, rlabel for every segment per patient. rhythm_data_generator() <- task_data_generator()

## Entire Function Workflow

load_datasets() -> load_train_datasets()/load_test_datasets() -> load_model()/create_model() (data is ready and you can simply feed the data into the model to receive the predicted label for every sample (frame_size = 400 / 100HZ == 4 seconds))

## Generator Associations

PatientGenerator (patient_id, patient_data with all segments under every patient): uniform_patient_generator() -> SampleGenerator: signal_generator()

## Test the pretrained model

Simply load the task hanlder and run evaluation with the proper params: 
```python
task="arrhythmia"
task_handler = TaskFactory.get(task)

# In demo we will cover 5 regions at a time, frame_size*5
config = 'configs/arrhythmia-100class-2.json'

task_handler.evaluate(parse_content(HKTestParams, config))
```

Below is the breakdown of the evaluation method class

```python
@staticmethod
def evaluate(params: HKTestParams):
    """Evaluate model

    Args:
        params (HKTestParams): Evaluation parameters
    """
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    class_names = get_class_names(params.num_classes)
    class_map = get_class_mapping(params.num_classes)
    input_spec = (
        tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
        tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
    )

    datasets = load_datasets(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        sampling_rate=params.sampling_rate,
        class_map=class_map,
        spec=input_spec,
        datasets=params.datasets,
    )
    # attach frame label
    test_x, test_y = load_test_datasets(datasets=datasets, params=params)

    # this is where you load the model, model should be ./results/arrhythmia-100class-2/model.keras
    with tfmot.quantization.keras.quantize_scope():
        logger.info("Loading model")
        model = tfa.load_model(params.model_file)
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing inference")
        y_true = np.argmax(test_y, axis=-1) # this is getting the true label of every sample (frame_size==400) on the last axis 0
        y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=-1)
    # END WITH

    # Summarize results
    logger.info("Testing Results")
    test_acc = np.sum(y_pred == y_true) / len(y_true)
    test_f1 = f1_score(y_true, y_pred, average="weighted")
    test_precision = precision_score(y_true, y_pred, average="weighted")
    test_recall = recall_score(y_true, y_pred, average="weighted")
    test_auc = roc_auc_score(y_true, y_pred)
    logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}, Precision={test_precision:.2%}, Sensitivity={test_recall: .2%}, AUC-ROC={test_auc: .2%}")

    if params.num_classes == 2:
        roc_path = params.job_dir / "roc_auc_test.png"
        roc_auc_plot(y_true, y_prob[:, 1], labels=class_names, save_path=roc_path)
    # END IF

    # If threshold given, only count predictions above threshold
    if params.threshold:
        prev_numel = len(y_true)
        y_prob, y_pred, y_true = threshold_predictions(y_prob, y_pred, y_true, params.threshold)
        drop_perc = 1 - len(y_true) / prev_numel
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        logger.info(f"[TEST SET] THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
    # END IF

    cm_path = params.job_dir / "confusion_matrix_test.png"
    confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")

```