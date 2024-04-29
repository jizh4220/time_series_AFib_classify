import datetime
import logging
import os
import random
import shutil

import keras
import numpy as np
import plotly.graph_objects as go
import sklearn.utils
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from plotly.subplots import make_subplots
from tqdm import tqdm
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


from ... import tflite as tfa
from ...defines import HKDemoParams, HKExportParams, HKTestParams, HKTrainParams
from ...metrics import confusion_matrix_plot, roc_auc_plot
from ...models.utils import threshold_predictions
from ...rpc.backends import EvbBackend, PcBackend
from ...utils import env_flag, set_random_seed, setup_logger
from ..task import HKTask
from .defines import (
    get_class_mapping,
    get_class_names,
    get_class_shape,
    get_classes,
    get_feat_shape,
)
from .utils import (
    create_model,
    load_datasets,
    load_test_datasets,
    load_train_datasets,
    prepare,
)

from enum import IntEnum
from heartkit.defines import HeartRhythm

class IcentiaRhythm(IntEnum):
    """Icentia rhythm labels"""

    noise = 0
    normal = 1
    afib = 2
    aflut = 3
    end = 4

HeartRhythmMap = {
    IcentiaRhythm.noise: HeartRhythm.noise,
    IcentiaRhythm.normal: HeartRhythm.normal,
    IcentiaRhythm.afib: HeartRhythm.afib,
    IcentiaRhythm.aflut: HeartRhythm.aflut,
    IcentiaRhythm.end: HeartRhythm.noise,
}

logger = setup_logger(__name__)


class AFIB_Ident(HKTask):
    """Xinio AFIB Identification Task"""

    @staticmethod
    def train(params: HKTrainParams):
        """Train  model

        Args:
            params (HKTrainParams): Training parameters
        """
        params.seed = set_random_seed(params.seed)
        logger.info(f"Random seed {params.seed}")

        os.makedirs(params.job_dir, exist_ok=True)
        logger.info(f"Creating working directory in {params.job_dir}")

        handler = logging.FileHandler(params.job_dir / "train.log", mode="w")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        with open(params.job_dir / "train_config.json", "w", encoding="utf-8") as fp:
            fp.write(params.model_dump_json(indent=2))

        if env_flag("WANDB"):
            wandb.init(
                project=f"hk-arrhythmia-{params.num_classes}",
                entity="ambiq",
                dir=params.job_dir,
            )
            wandb.config.update(params.model_dump())
        # END IF

        classes = get_classes(params.num_classes)
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
            spec=input_spec,
            class_map=class_map,
            datasets=params.datasets,
        )
        train_ds, val_ds = load_train_datasets(datasets=datasets, params=params)

        test_labels = [label.numpy() for _, label in val_ds]
        y_true = np.argmax(np.concatenate(test_labels), axis=-1)

        class_weights = 0.25
        if params.class_weights == "balanced":
            class_weights = sklearn.utils.compute_class_weight("balanced", classes=np.array(classes), y=y_true)

        with tfa.get_strategy().scope():
            inputs = keras.Input(shape=input_spec[0].shape, batch_size=None, name="input", dtype=input_spec[0].dtype)
            if params.resume and params.model_file:
                logger.info(f"Loading model from file {params.model_file}")
                model = tfa.load_model(params.model_file)
            else:
                logger.info("Creating model from scratch")
                model = create_model(
                    inputs,
                    num_classes=params.num_classes,
                    architecture=params.architecture,
                )
            # END IF

            flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

            if params.lr_cycles > 1:
                scheduler = keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=params.lr_rate,
                    first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                    t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
                    m_mul=0.4,
                )
            else:
                scheduler = keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=params.lr_rate, decay_steps=params.steps_per_epoch * params.epochs
                )
            # END IF
            optimizer = keras.optimizers.Adam(scheduler)
            loss = keras.losses.CategoricalFocalCrossentropy(from_logits=True, alpha=class_weights)
            metrics = [
                keras.metrics.CategoricalAccuracy(name="acc"),
                tfa.MultiF1Score(name="f1", average="weighted"),
            ]

            if params.resume and params.weights_file:
                logger.info(f"Hydrating model weights from file {params.weights_file}")
                model.load_weights(params.weights_file)

            if params.model_file is None:
                
                params.model_file = params.job_dir / "model.keras"

            # Perform QAT if requested
            if params.quantization.enabled and params.quantization.qat:
                logger.info("Performing QAT")
                model = tfmot.quantization.keras.quantize_model(model, quantized_layer_name_prefix="q_")
            # END IF

            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            model(inputs)
            model.summary(print_fn=logger.info)
            logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

            ModelCheckpoint = keras.callbacks.ModelCheckpoint
            if env_flag("WANDB"):
                ModelCheckpoint = WandbModelCheckpoint
            model_callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor=f"val_{params.val_metric}",
                    patience=max(int(0.25 * params.epochs), 1),
                    mode="max" if params.val_metric == "f1" else "auto",
                    restore_best_weights=True,
                ),
                ModelCheckpoint(
                    filepath=str(params.model_file),
                    monitor=f"val_{params.val_metric}",
                    save_best_only=True,
                    mode="max" if params.val_metric == "f1" else "auto",
                    verbose=1,
                ),
                keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
            ]
            if env_flag("TENSORBOARD"):
                model_callbacks.append(
                    keras.callbacks.TensorBoard(
                        log_dir=params.job_dir,
                        write_steps_per_second=True,
                    )
                )
            if env_flag("WANDB"):
                model_callbacks.append(WandbMetricsLogger())

            # TODO: check the validation data loading here
            try:
                model.fit(
                    train_ds,
                    steps_per_epoch=params.steps_per_epoch,
                    verbose=2,
                    epochs=params.epochs,
                    validation_data=val_ds,
                    callbacks=model_callbacks,
                )
            except KeyboardInterrupt:
                logger.warning("Stopping training due to keyboard interrupt")

            logger.info(f"Model saved to {params.model_file}")

            # Get full validation results
            keras.models.load_model(params.model_file)
            logger.info("Performing full validation")
            y_pred = np.argmax(model.predict(val_ds), axis=-1)

            cm_path = params.job_dir / "confusion_matrix.png"
            confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
            if env_flag("WANDB"):
                conf_mat = wandb.plot.confusion_matrix(preds=y_pred, y_true=y_true, class_names=class_names)
                wandb.log({"conf_mat": conf_mat})
            # END IF

            # Summarize results
            test_acc = np.sum(y_pred == y_true) / len(y_true)
            test_f1 = f1_score(y_true, y_pred, average="weighted")
            test_precision = precision_score(y_true, y_pred, average="weighted")
            test_recall = recall_score(y_true, y_pred, average="weighted")
            test_auc = roc_auc_score(y_true, y_pred)
            logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}, Precision={test_precision:.2%}, Sensitivity={test_recall: .2%}, AUC-ROC={test_auc: .2%}")
        # END WITH

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

    @staticmethod
    def export(params: HKExportParams):
        """Export model

        Args:
            params (HKExportParams): Deployment parameters
        """

        os.makedirs(params.job_dir, exist_ok=True)
        logger.info(f"Creating working directory in {params.job_dir}")

        handler = logging.FileHandler(params.job_dir / "export.log", mode="w")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        tfl_model_path = params.job_dir / "model.tflite"
        tflm_model_path = params.job_dir / "model_buffer.h"

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
        test_x, test_y = load_test_datasets(datasets=datasets, params=params)

        # Load model and set fixed batch size of 1
        logger.info("Loading trained model")
        with tfmot.quantization.keras.quantize_scope():
            model = tfa.load_model(params.model_file)

        inputs = keras.Input(shape=input_spec[0].shape, batch_size=1, name="input", dtype=input_spec[0].dtype)
        outputs = model(inputs)

        if not params.use_logits and not isinstance(model.layers[-1], keras.layers.Softmax):
            outputs = keras.layers.Softmax()(outputs)
            model = keras.Model(inputs, outputs, name=model.name)
            outputs = model(inputs)
        # END IF

        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info(f"Converting model to TFLite (quantization={params.quantization.enabled})")
        if params.quantization.enabled:
            _, quant_df = tfa.debug_quant_tflite(
                model=model,
                test_x=test_x,
                input_type=params.quantization.input_type,
                output_type=params.quantization.output_type,
                supported_ops=params.quantization.supported_ops,
            )
            quant_df.to_csv(params.job_dir / "quant.csv")
        # END IF

        tflite_model = tfa.convert_tflite(
            model=model,
            quantize=params.quantization.enabled,
            test_x=test_x,
            input_type=params.quantization.input_type,
            output_type=params.quantization.output_type,
            supported_ops=params.quantization.supported_ops,
        )

        # Save TFLite model
        logger.info(f"Saving TFLite model to {tfl_model_path}")
        with open(tfl_model_path, "wb") as fp:
            fp.write(tflite_model)

        # Save TFLM model
        logger.info(f"Saving TFL micro model to {tflm_model_path}")
        tfa.xxd_c_dump(
            src_path=tfl_model_path,
            dst_path=tflm_model_path,
            var_name=params.tflm_var_name,
            chunk_len=20,
            is_header=True,
        )

        # Verify TFLite results match TF results
        logger.info("Validating model results")
        y_true = np.argmax(test_y, axis=-1)
        y_pred_tf = np.argmax(model.predict(test_x), axis=-1)
        y_pred_tfl = np.argmax(tfa.predict_tflite(model_content=tflite_model, test_x=test_x), axis=-1)

        test_acc = np.sum(y_pred_tf == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred_tf, average="weighted")
        test_precision = precision_score(y_true, y_pred_tf, average="weighted")
        test_recall = recall_score(y_true, y_pred_tf, average="weighted")
        test_auc = roc_auc_score(y_true, y_pred_tf)
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}, Precision={test_precision:.2%}, Sensitivity={test_recall: .2%}, AUC-ROC={test_auc: .2%}")
        tf_acc = test_acc

        tfl_acc = np.sum(y_true == y_pred_tfl) / y_true.size
        tfl_f1 = f1_score(y_true, y_pred_tfl, average="weighted")
        logger.info(f"[TFL SET] ACC={tfl_acc:.2%}, F1={tfl_f1:.2%}")

        # Check accuracy hit
        tfl_acc_drop = max(0, tf_acc - tfl_acc)
        if params.val_acc_threshold is not None and (1 - tfl_acc_drop) < params.val_acc_threshold:
            logger.warning(f"TFLite accuracy dropped by {tfl_acc_drop:0.2%}")
        elif params.val_acc_threshold:
            logger.info(f"Validation passed ({tfl_acc_drop:0.2%})")

        if params.tflm_file and tflm_model_path != params.tflm_file:
            logger.info(f"Copying TFLM header to {params.tflm_file}")
            shutil.copyfile(tflm_model_path, params.tflm_file)
        
        # we also want to get all the mislablled results here
        # Get the indices of the mislabeled data
        mislabeled_indices = np.where(y_pred_tf != y_true)[0]
        # Extract the mislabeled data
        mislabeled_data = test_x[mislabeled_indices]
        np.save(params.job_dir / 'mislabeled_data.npy', mislabeled_data)
        # Extract the true labels of the mislabeled data
        true_labels = y_true[mislabeled_indices]
        np.save(params.job_dir / 'true_labels.npy', true_labels)
        # Extract the predicted labels of the mislabeled data
        predicted_labels = y_pred_tf[mislabeled_indices]
        np.save(params.job_dir / 'predicted_labels.npy', predicted_labels)
        # true_labels_reshaped = np.reshape(true_labels, (true_labels.shape[0], 1, 1, 1))
        # predicted_labels_reshaped = np.reshape(predicted_labels, (predicted_labels.shape[0], 1, 1, 1))

        # # Add an extra dimension to mislabeled_data
        # mislabeled_data_expanded = np.expand_dims(mislabeled_data, axis=1)

        # # Now concatenate the arrays along the second dimension (axis=1)
        # expanded_data = np.concatenate((mislabeled_data_expanded, true_labels_reshaped, predicted_labels_reshaped), axis=1)
        # np.save(params.job_dir / 'expanded_data.npy', expanded_data)

        # visualize
        # import matplotlib.pyplot as plt
        # from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, precision_recall_curve, auc

        # # Assuming 'model' is your trained model and 'X_test' is your test data
        # fig, ax = plt.subplots()

        # Confusion Matrix
        # plot_confusion_matrix(model, test_x, y_true, ax=ax, normalize='true')
        # ax.set_title('Confusion Matrix')
        # plt.show()

        # ROC Curve
        # plot_roc_curve(model, test_x, y_true, ax=ax)
        # ax.set_title('ROC Curve')
        # plt.show()

        # # Precision-Recall Curve
        # precision, recall, _ = precision_recall_curve(y_true, y_pred)
        # auc_score = auc(recall, precision)
        # plt.plot(recall, precision, marker='.')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve - AUC={auc_score:.2f}')
        # plt.show()


    @staticmethod
    def demo(params: HKDemoParams):
        """Run demo for model

        Args:
            params (HKDemoParams): Demo parameters
        """

        bg_color = "rgba(38,42,50,1.0)"
        primary_color = "#11acd5"
        secondary_color = "#ce6cff"
        third_color = "#a1d34f"  # A shade of green
        fourth_color = "#f5e642"  # A shade of yellow
        plotly_template = "plotly_dark"

        color_dict = {
            0: "#11acd5",  # Blue color for 0
            1: "#ce6cff",  # Purple color for 1
            2: "#a1d34f"   # Green color for 2
        }
        # Load backend inference engine
        BackendRunner = EvbBackend if params.backend == "evb" else PcBackend
        runner = BackendRunner(params=params)

        # Load data
        class_names = get_class_names(params.num_classes)
        class_map = get_class_mapping(params.num_classes)
        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        tgt_map = {k: class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}

        # this will be one patient five frames?
        ds = load_datasets(
            ds_path=params.ds_path,
            frame_size=5 * params.frame_size,
            sampling_rate=params.sampling_rate,
            class_map=class_map,
            spec=input_spec,
            datasets=params.datasets,
        )[0]

        patient_ids = ds.get_test_patient_ids()
        # this is where the signal being extracted by the first patient_id, patient_data
        signal_label = next(ds.signal_label_generator(ds.uniform_patient_generator(patient_ids=patient_ids, repeat=False)))
        x = signal_label[0]
        # y_sig  = signal_label[1]
        y_sig = signal_label[1]
        segment_id = signal_label[2]
        patient_id = patient_ids[0]
        
        logger.info(f"Current patient {patient_id}, segment_id is {segment_id}")
        runner.open()
        logger.info("Running inference")
        # assume all zeros in the beginning
        y_pred = np.zeros(x.shape[0], dtype=np.int32)
        y_sig = y_sig[np.where(~np.isin(y_sig[:, 1], [IcentiaRhythm.noise.value, IcentiaRhythm.end.value]))] # filter the noise and end
        print(y_sig)
        y_orig = np.vectorize(tgt_map.get, otypes=[int])(y_sig[0::2, 1]) # from 0-4 to 0-3
        print(y_orig)
        if len(y_orig) == 0:
            y_orig = np.full(x.shape[0], 1)
        if len(y_orig) == 1:
            y_orig = np.full(x.shape[0], y_orig[0])
        else: # a more complicated cases where you have AFib mixed with AFlut
            # let's do majority voting here
            #TODO majority voting for this multi-peak case
            print("Majority voting for multi-rlabel case")
            y_orig = np.full(x.shape[0], np.argmax(np.bincount(y_orig)))
            # y_delta = y_sig[:, 0][1] - y_sig[:, 0][0] # the middle frame between AFib and AFlut
            # y_orig = np.full(x.shape[0], y_orig[0])
            # if y_sig[:, 0][0] < x.shape[0] - y_sig[:, 0][1]:
            #     y_orig = np.full(x.shape[0],  y_sig[:, 1][1])
            # else:
            #     y_orig = np.full(x.shape[0],  y_sig[:, 1][0])


        # plot the prediction label inside the for loop
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[
                [{"colspan": 1, "type": "xy", "secondary_y": True}],
            ],
            subplot_titles=(None, None),
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )

        tod = datetime.datetime(2024, 5, 24, random.randint(12, 23), 00)
        ts = np.array([tod + datetime.timedelta(seconds=i / params.sampling_rate) for i in range(x.shape[0])])
        # print(f"Size of y_orig{len(y_orig)}, and rlabels of {y_orig}")
        # 2000/400 = 5
        for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
            # this is [x.shape[0] - 400, x.shape[0]], get the earlier peak, this is the end
            if i + params.frame_size > x.shape[0]:
                start, stop = x.shape[0] - params.frame_size, x.shape[0]
            else:
                start, stop = i, i + params.frame_size
            # print("Before inference this is the ts:", i, start, stop)
            xx = prepare(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
            runner.set_inputs(xx)
            runner.perform_inference()
            yy = runner.get_outputs()
            # y_orig[start:stop] = 
            # this is the predicted label for current frame
            y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
            # print(ts[start], ts[stop-1] - datetime.timedelta(seconds=0.1), y_pred[start])


            fig.add_annotation(
                x=ts[start] + (ts[stop-1] - ts[start]) / 2,
                y=np.min(x),
                text=class_names[y_pred[start]],
                showarrow=False,
                font=dict(color=color_dict[y_pred[start]]),
            )

            # predicted results
            fig.add_vrect(
                x0=ts[start],
                x1=ts[stop-1] - datetime.timedelta(seconds=0.1),
                y0=0,
                # y1=np.max(x[start:stop]) / 2,
                y1=np.min(x),  
                # annotation_text=class_names[y_pred[start]],
                # annotation_position="inside top",
                fillcolor=color_dict[y_pred[start]],
                opacity=0.25,
                line_width=0,
                row=1,
                col=1,
                secondary_y=False,
            )

            if y_orig[i // params.frame_size] == -1:
                print("Unknown Peak should be labeled here")
                orig_label = "Unknown"
            else:
                orig_label = class_names[y_orig[i // params.frame_size]]
            # original results
            fig.add_annotation(
                x=ts[start] + (ts[stop-1] - ts[start]) / 2,
                y=np.max(x),
                text=orig_label,
                showarrow=False,
                font=dict(color=color_dict[y_orig[i // params.frame_size]]),
            )

            fig.add_vrect(
                x0=ts[start],
                x1=ts[stop-1] - datetime.timedelta(seconds=0.1),
                y0=np.max(x)/2,
                # y1=np.max(x[start:stop]) / 2,
                y1=np.max(x)*0.8,  
                # annotation_text=class_names[y_pred[start]],
                # annotation_position="inside top",
                fillcolor=color_dict[y_orig[i // params.frame_size]],
                opacity=0.25,
                line_width=0,
                row=1,
                col=1,
                secondary_y=False,
            )

            # predction != original
            if y_pred[start] != y_orig[i // params.frame_size]:
                fig.add_vrect(
                    x0=ts[start],
                    x1=ts[stop-1] - datetime.timedelta(seconds=0.1),
                    y0=0,
                    y1=np.max(x)*0.8,  # Upper limit of the rectangle
                    # annotation_text=class_names[y_pred[start]],
                    fillcolor="red",
                    opacity=0.25,
                    line_width=2,
                    line_color="red",
                    row=1,
                    col=1,
                    secondary_y=False,
                )
                # fig.add_annotation(x=ts[stop-1] - datetime.timedelta(seconds=2.5), y=np.max(x), 
                #     text=str(y_orig[i // 400]), showarrow=True, font=dict(color='green'))
            
        # END FOR
        runner.close()

        # Report
        logger.info("Generating report")

        # print(f"{len(y_pred)} and {len(ts)}")
        pred_bounds = np.concatenate(([0], np.diff(y_pred).nonzero()[0] + 1, [y_pred.size - 1]))

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=x,
                name="ECG",
                mode="lines",
                line=dict(color=primary_color, width=2),
                showlegend=False,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
            
        # logger.info(f"How many frames and labels we will have: {y_orig} and how many predictions we will have: {y_pred}")
        # logger.info(f"{range(1, len(pred_bounds))}")
        for i in range(1, len(pred_bounds)):
            start, stop = pred_bounds[i - 1], pred_bounds[i]
            pred_class = y_pred[start]
            mid_point = start + (stop - start) // 2  # Calculate mid point of each bound

            # fig.add_vrect(
            #     x0=ts[start],
            #     x1=ts[stop],
            #     y0=x/2,
            #     y1=x,  # Upper limit of the rectangle
            #     annotation_text=y_orig[i-1],
            #     fillcolor=secondary_color,
            #     opacity=0.25,
            #     line_width=2,
            #     line_color=secondary_color,
            #     row=1,
            #     col=1,
            #     secondary_y=False,
            # )
            if pred_class <= 0:
                continue
            # logger.info(f"Current true label: {y_orig[i-1]}")

            # adding the AFIB prediction detection here
            # fig.add_vrect(
            #     x0=ts[start],
            #     x1=ts[stop],
            #     y0=np.max(x)/2,
            #     y1=np.max(x)*0.8,  # Upper limit of the rectangle
            #     annotation_text=class_names[pred_class],
            #     fillcolor=third_color,
            #     opacity=0.25,
            #     line_width=2,
            #     line_color=third_color,
            #     row=1,
            #     col=1,
            #     secondary_y=False,
            # )

        fig.update_layout(
            template=plotly_template,
            height=800,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            margin=dict(l=10, r=10, t=80, b=80),
            legend=dict(groupclick="toggleitem"),
            title=f"Patient ID: {patient_id}, Segment ID: {segment_id}",
            title_x=0.5,
            # title_y=0.9,
            # title_font=dict(size=20),
            # annotations=[
            #     dict(
            #         text=f"Patient ID: {patient_id}, Segment ID: {segment_id}",
            #         showarrow=False,
            #         xref='paper',
            #         yref='paper',
            #         x=0.5,
            #         y=0.9,
            #         font=dict(size=14)
            #     )
            # ]
        )
        
        fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=True)
        fig.show()

        logger.info(f"Report saved to {params.job_dir / 'demo.html'}")


    #TODO: longer demo for the model
    @staticmethod
    def longer_demo(params: HKDemoParams, n_sample=15):
        """Run demo for model

        Args:
            params (HKDemoParams): Demo parameters
        """

        bg_color = "rgba(38,42,50,1.0)"
        primary_color = "#11acd5"
        plotly_template = "plotly_dark"

        color_dict = {
            -1: "#505050",  # Grey color for -1
            0: "#11acd5",  # Blue color for 0
            1: "#ce6cff",  # Purple color for 1
            2: "#a1d34f"   # Green color for 2
        }
        # Load backend inference engine
        BackendRunner = EvbBackend if params.backend == "evb" else PcBackend
        runner = BackendRunner(params=params)

        # Load data
        class_names = get_class_names(params.num_classes)
        class_map = get_class_mapping(params.num_classes)
        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        tgt_map = {k: class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
        nrow = int(n_sample/5)
        row_idx = 0
        # this will be one patient five frames?
        ds = load_datasets(
            ds_path=params.ds_path,
            frame_size= n_sample * params.frame_size,
            sampling_rate=params.sampling_rate,
            class_map=class_map,
            spec=input_spec,
            datasets=params.datasets,
        )[0]

        patient_ids = ds.get_test_patient_ids()
        # this is where the signal being extracted by the first patient_id, patient_data
        signal_label = next(ds.signal_label_generator(ds.uniform_patient_generator(patient_ids=patient_ids, repeat=False)))
        x = signal_label[0]
        # y_sig  = signal_label[1]
        y_sig = signal_label[1]
        segment_id = signal_label[2]
        patient_id = patient_ids[0]
        
        logger.info(f"Current patient {patient_id}, segment_id is {segment_id}")
        runner.open()
        logger.info("Running inference")
        # assume all zeros in the beginning
        y_pred = np.zeros(x.shape[0], dtype=np.int32)
        print(f"Pre-filter QC: {y_sig}")
        y_sig = y_sig[np.where(~np.isin(y_sig[:, 1], [IcentiaRhythm.noise.value, IcentiaRhythm.end.value]))] # filter the noise and end
        y_orig = np.vectorize(tgt_map.get, otypes=[int])(y_sig[:, 1]) # from 0-4 to 0-3
        print(y_orig)
        if len(y_orig) == 0:
            print("Unidentified label")
            y_orig = np.full(x.shape[0], -1)
        elif len(y_orig) == 1:
            y_orig = np.full(x.shape[0], y_orig[0])
        else: # a more complicated cases where you have AFib mixed with AFlut
            # let's do majority voting here
            print("Majority voting for multi-rlabel case")
            y_orig = np.full(x.shape[0], np.argmax(np.bincount(y_orig)))


        logger.info("Start generating report on ECG prediction")
        # plot the prediction label inside the for loop
        fig = make_subplots(
            rows=nrow,
            cols=1,
            specs=[[{"colspan": 1, "type": "xy", "secondary_y": True}]] * nrow,
            subplot_titles=(None, None),
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )

        tod = datetime.datetime(2024, 5, 24, random.randint(12, 23), 00)
        ts = np.array([tod + datetime.timedelta(seconds=i / params.sampling_rate) for i in range(x.shape[0])])
        # segment_length = 5*params.frame_size
        # print(f"Size of y_orig{len(y_orig)}, and rlabels of {y_orig}")
        # 2000/400 = 5
        for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
            if i % (5*params.frame_size) == 0:
                #ts = np.array([tod + datetime.timedelta(seconds=i-row_idx*5*params.frame_size / params.sampling_rate) for i in range(x.shape[0])])
                # start a new row for the make_plots
                row_idx += 1
                print(row_idx)
            # this is [x.shape[0] - 400, x.shape[0]], get the earlier peak, this is the end
            if i + params.frame_size > x.shape[0]:
                start, stop = x.shape[0] - params.frame_size, x.shape[0]
            else:
                start, stop = i, i + params.frame_size
            # print("Before inference this is the ts:", i, start, stop)
            xx = prepare(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
            runner.set_inputs(xx)
            runner.perform_inference()
            yy = runner.get_outputs()
            # y_orig[start:stop] = 
            # this is the predicted label for current frame
            y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
            # print(ts[start], ts[stop-1] - datetime.timedelta(seconds=0.1), y_pred[start])


            fig.add_annotation(
                x=ts[start] + (ts[stop-1] - ts[start]) / 2,
                y=np.min(x)*0.8,
                text=class_names[y_pred[start]],
                showarrow=False,
                row=row_idx,
                col=1,
                font=dict(color=color_dict[y_pred[start]]),
            )

            # predicted results
            fig.add_vrect(
                x0=ts[start],
                x1=ts[stop-1] - datetime.timedelta(seconds=0.1),
                y0=np.min(x)/3+0.2,
                # y1=np.max(x[start:stop]) / 2,
                y1=np.min(x)/3,  
                fillcolor=color_dict[y_pred[start]],
                opacity=0.25,
                line_width=0,
                row=row_idx,
                col=1,
                secondary_y=False,
            )


            # original results
            fig.add_annotation(
                x=ts[start] + (ts[stop-1] - ts[start]) / 2,
                y=np.max(x),
                text=class_names[y_orig[i // 400]],
                showarrow=False,
                row=row_idx,
                col=1,
                font=dict(color=color_dict[y_orig[i // 400]]),
            )

            fig.add_vrect(
                x0=ts[start],
                x1=ts[stop-1] - datetime.timedelta(seconds=0.1),
                # y0=np.max(x)/2,
                y0=0.9,
                # y1=np.max(x)*0.8,  
                y1=1.1,
                # annotation_text=class_names[y_pred[start]],
                # annotation_position="inside top",
                fillcolor=color_dict[y_orig[i // 400]],
                opacity=0.25,
                line_width=0,
                row=row_idx,
                col=1,
                secondary_y=False,
            )

            # predction != original
            if y_pred[start] != y_orig[i // 400]:
                logger.info(f"Catch contradctions!")
                fig.add_vrect(
                    x0=ts[start],
                    x1=ts[stop-1] - datetime.timedelta(seconds=0.1),
                    # y0=np.max(x)/2,
                    y0=0.9,
                    # y1=np.max(x)*0.8,  
                    y1=1.1,
                    # annotation_text=class_names[y_pred[start]],
                    fillcolor="red",
                    opacity=0.25,
                    line_width=2,
                    line_color="red",
                    row=row_idx,
                    col=1,
                    secondary_y=False,
                )

            fig.add_trace(
                go.Scatter(
                    x=ts[start:stop],
                    y=x[start:stop],
                    name="ECG",
                    mode="lines",
                    line=dict(color=primary_color, width=2),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
                secondary_y=False,
            )            
        # END FOR
        runner.close()

        # Report
        logger.info("Generating report")

        # fig.add_trace(
        #     go.Scatter(
        #         x=ts,
        #         y=x,
        #         name="ECG",
        #         mode="lines",
        #         line=dict(color=primary_color, width=2),
        #         showlegend=False,
        #     ),
        #     row=row_idx,
        #     col=1,
        #     secondary_y=False,
        # )
            

        fig.update_layout(
            template=plotly_template,
            height=400*nrow,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            margin=dict(l=10, r=10, t=80, b=80),
            legend=dict(groupclick="toggleitem"),
            title=f"Patient ID: {patient_id}, Segment ID: {segment_id}",
            title_x=0.5,
        )
        
        fig.write_html(params.job_dir / "longer_demo.html", include_plotlyjs="cdn", full_html=True)
        fig.show()

        logger.info(f"Report saved to {params.job_dir / 'longer_demo.html'}")


    @staticmethod
    def continous_minute_evaluation(params: HKDemoParams, pid: int, sid: int, start: int, end: int):
        bg_color = "rgba(38,42,50,1.0)"
        primary_color = "#11acd5"
        plotly_template = "plotly_dark"

        color_dict = {
            -1: "#505050",  # Grey color for -1
            0: "#11acd5",  # Blue color for 0
            1: "#ce6cff",  # Purple color for 1
            2: "#a1d34f"   # Green color for 2
        }
        # Load backend inference engine
        BackendRunner = EvbBackend if params.backend == "evb" else PcBackend
        runner = BackendRunner(params=params)

        # Load data
        class_names = get_class_names(params.num_classes)
        class_map = get_class_mapping(params.num_classes)
        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        tgt_map = {k: class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
        row_idx = 0

        hour_frame = 15 * 60 #one minute * 60
        ds = load_datasets(
            ds_path=params.ds_path,
            frame_size=hour_frame * n_hour * params.frame_size,
            sampling_rate=params.sampling_rate,
            class_map=class_map,
            spec=input_spec,
            datasets=params.datasets,
        )[0]

        patient_ids = ds.get_test_patient_ids()
        BackendRunner = EvbBackend if params.backend == "evb" else PcBackend
        runner = BackendRunner(params=params)

        single_pat_gen = ds.uniform_patient_generator(patient_ids=patient_ids[patient_ids in pid], repeat=False, shuffle=False)
        continuous_gen = ds.continous_signal_label_generator(single_pat_gen, 240) # since this is larger than 60, we will have multiple segments together
        continuous_pat = next(continuous_gen)
        whole_seg_pred = []
        for i in range(len(continuous_pat)):
            x = continuous_pat[i]['x']
            # y_sig  = signal_label[1]
            y_sig = continuous_pat[i]['frame_rlabel']
            segment_id = continuous_pat[i]['segment_id']
            y_pred = np.zeros(x.shape[0], dtype=np.int32)
            runner.open()

            tgt_labels = list(set(class_map.values()))
            class_map = get_class_mapping(params.num_classes)
            tgt_map = {k: class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
            y_sig = y_sig[np.where(~np.isin(y_sig[:, 1], [IcentiaRhythm.noise.value, IcentiaRhythm.end.value]))] # filter the noise and end
            y_orig = np.vectorize(tgt_map.get, otypes=[int])(y_sig[:, 1]) # from 0-4 to 0-3
            print(y_orig)
            if len(y_orig) == 0:
                print("Unidentified label")
                y_orig = np.full(x.shape[0], -1)
            elif len(y_orig) == 1:
                y_orig = np.full(x.shape[0], y_orig[0])
            else: # a more complicated cases where you have AFib mixed with AFlut
                # let's do majority voting here
                print("Majority voting for multi-rlabel case")
                y_orig = np.full(x.shape[0], np.argmax(np.bincount(y_orig)))
                
            tod = datetime.datetime(2024, 5, 24, random.randint(12, 23), 00)
            ts = np.array([tod + datetime.timedelta(seconds=i / params.sampling_rate) for i in range(x.shape[0])])
            # segment_length = 5*params.frame_size
            # print(f"Size of y_orig{len(y_orig)}, and rlabels of {y_orig}")
            # 2000/400 = 5
            row_idx = 0
            ratios = []
            for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
                if i % (5*params.frame_size) == 0:
                    #ts = np.array([tod + datetime.timedelta(seconds=i-row_idx*5*params.frame_size / params.sampling_rate) for i in range(x.shape[0])])
                    # start a new row for the make_plots
                    row_idx += 1
                    print(row_idx)
                # this is [x.shape[0] - 400, x.shape[0]], get the earlier peak, this is the end
                if i + params.frame_size > x.shape[0]:
                    start, stop = x.shape[0] - params.frame_size, x.shape[0]
                else:
                    start, stop = i, i + params.frame_size

                # print("Before inference this is the ts:", i, start, stop)
                xx = prepare(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
                runner.set_inputs(xx)
                runner.perform_inference()
                yy = runner.get_outputs()
                # y_orig[start:stop] = 
                # this is the predicted label for current frame
                y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
                # Assuming y_pred and y_orig are numpy arrays
                if y_pred[i] == y_orig[i]:
                    ratios.append(1)
                else:
                    ratios.append(0)

            print(np.sum(ratios) / (len(y_pred) / 400))
            whole_seg_pred.append(ratios)
        NotImplementedError

        