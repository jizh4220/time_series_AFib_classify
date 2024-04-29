import functools
import logging
import os
import random
import tempfile
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import IntEnum
from multiprocessing import Pool

import boto3
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import physiokit as pk
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm
from typing import List, Tuple, Dict


from ..defines import HeartBeat, HeartRate, HeartRhythm, HeartSegment
from ..utils import download_file
from .dataset import HKDataset
from .defines import PatientGenerator, SampleGenerator

logger = logging.getLogger(__name__)


class IcentiaRhythm(IntEnum):
    """Icentia rhythm labels"""

    noise = 0
    normal = 1
    afib = 2
    aflut = 3
    end = 4

class IcentiaQuality(IntEnum):
    """Icentia rhythm labels"""
    noise = 0
    quality = 1


class IcentiaBeat(IntEnum):
    """Incentia beat labels"""

    undefined = 0
    normal = 1
    pac = 2
    aberrated = 3
    pvc = 4


class IcentiaHeartRate(IntEnum):
    """Icentia heart rate labels"""

    tachycardia = 0
    bradycardia = 1
    normal = 2
    noise = 3


# These map Icentia specific labels to common labels
HeartRhythmMap = {
    IcentiaRhythm.noise: HeartRhythm.noise,
    IcentiaRhythm.normal: HeartRhythm.normal,
    IcentiaRhythm.afib: HeartRhythm.afib,
    IcentiaRhythm.aflut: HeartRhythm.aflut,
    IcentiaRhythm.end: HeartRhythm.noise,
}

HeartBeatMap = {
    IcentiaBeat.undefined: HeartBeat.noise,
    IcentiaBeat.normal: HeartBeat.normal,
    IcentiaBeat.pac: HeartBeat.pac,
    IcentiaBeat.aberrated: HeartBeat.pac,
    IcentiaBeat.pvc: HeartBeat.pvc,
}


def get_class_mapping(nclasses: int = 2) -> dict[int, int]:
    """Get class mapping

    Args:
        nclasses (int): Number of classes

    Returns:
        dict[int, int]: Class mapping
    """
    match nclasses:
        case 2:
            return {
                HeartRhythm.normal: 0,
                HeartRhythm.afib: 1,
                HeartRhythm.aflut: 1,
            }
        case 3:
            return {
                HeartRhythm.normal: 0,
                HeartRhythm.afib: 1,
                HeartRhythm.aflut: 2,
            }
        case _:
            raise ValueError(f"Invalid number of classes: {nclasses}")
        

def extract_label_data(segments: Dict, tgt_map: Dict, tgt_labels: List[int], input_size: int) -> List[np.ndarray]:
    """
    This function extracts label data from segments based on target labels, a mapping of labels to target labels, 
    and the input size.

    Args:
        segments (Dict): A dictionary of segments.
        tgt_map (Dict): A dictionary mapping labels to target labels.
        tgt_labels (List[int]): A list of target labels.
        input_size (int): The size of the input.

    Returns:
        pt_tgt_seg_map (List[np.ndarray]): A list of numpy arrays, where each array contains the segment index, 
        the start and end frames, and the target class for each target label.
    """

    # Map segment index to segment key
    seg_map: List[str] = list(segments.keys())

    # Initialize an empty list for each target label
    pt_tgt_seg_map = [[] for _ in tgt_labels]

    # Loop over each segment
    for seg_idx, seg_key in enumerate(seg_map):
        # Extract rhythm labels
        rlabels = segments[seg_key]["rlabels"][:]

        # Skip if no rhythm labels or only noise
        if not rlabels.shape[0] or not rlabels[np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)].shape[0]:
            continue

        # Unpack start, end, and label
        xs, xe, xl = rlabels[0::2, 0], rlabels[1::2, 0], rlabels[0::2, 1]

        # Map labels to target labels
        xl = np.vectorize(tgt_map.get, otypes=[int])(xl)

        # Capture segment, start, and end for each target label
        for tgt_idx, tgt_class in enumerate(tgt_labels):
            # Find frames large enough that contain the current class of label
            idxs = np.where((xe - xs >= input_size) & (xl == tgt_class))

            # Create an array filled with the segment index, followed by start and end indices of segment
            seg_vals = np.vstack((seg_idx * np.ones_like(idxs), xs[idxs], xe[idxs])).T

            # Append the segment values to the corresponding target label list
            pt_tgt_seg_map[tgt_idx] += seg_vals.tolist()

    # Convert each list in pt_tgt_seg_map to a numpy array
    pt_tgt_seg_map = [np.array(b) for b in pt_tgt_seg_map]

    return pt_tgt_seg_map

def generate_segment_samples(tgt_labels: List[int], pt_tgt_seg_map: dict, samples_per_tgt: List[int], input_size: int, if_random=True) -> List[Tuple[int, int, int, int]]:
    """
    This function generates segment samples based on target labels, a mapping of target indices to segments, 
    the number of samples per target, and the input size.

    Args:
        tgt_labels (List[int]): A list of target labels.
        pt_tgt_seg_map (dict): A dictionary mapping target indices to their corresponding segments.
        samples_per_tgt (List[int]): A list of the number of samples for each target.
        input_size (int): The size of the input.

    Returns:
        seg_samples (List[Tuple[int, int, int, int]]): A list of tuples, where each tuple contains the segment index, 
        the start and end frames, and the target class.
    """

    # Initialize an empty list to store the segment samples
    seg_samples: List[Tuple[int, int, int, int]] = []

    # Loop over each target label
    for tgt_idx, tgt_class in enumerate(tgt_labels):
        # Get the corresponding segments for the current target
        tgt_segments = pt_tgt_seg_map[tgt_idx]

        # If there are no segments for the current target, skip to the next target
        if not tgt_segments.shape[0]:
            continue

        # Generate a list of indices for the target segments. The probability of selecting each index is proportional to the length of the corresponding segment.
        tgt_seg_indices: List[int] = random.choices(
            np.arange(tgt_segments.shape[0]),
            weights=tgt_segments[:, 2] - tgt_segments[:, 1], # proportional to the length of every segment
            k=samples_per_tgt[tgt_idx],
        )

        # Loop over each selected segment index
        for tgt_seg_idx in tgt_seg_indices:
            # Get the segment index and the start and end frames of the rhythm
            seg_idx, rhy_start, rhy_end = tgt_segments[tgt_seg_idx]

            # Randomly select a frame within the segment sample
            frame_start = np.random.randint(rhy_start, rhy_end - input_size + 1)

            # Calculate the end frame based on the input size
            frame_end = frame_start + input_size

            # Append the segment sample to the list
            seg_samples.append((seg_idx, frame_start, frame_end, tgt_class))

    # Return the list of segment samples
    return seg_samples


class IcentiaDataset(HKDataset):
    """Icentia dataset"""

    def __init__(
        self,
        ds_path: os.PathLike,
        task: str,
        frame_size: int,
        target_rate: int,
        spec: tuple[tf.TensorSpec, tf.TensorSpec],
        class_map: dict[int, int] | None = None,
    ) -> None:
        super().__init__(
            ds_path=ds_path / "icentia11k",
            task=task,
            frame_size=frame_size,
            target_rate=target_rate,
            spec=spec,
            class_map=class_map,
        )

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 250

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0.0018

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1.3711

    @property
    def patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return np.arange(11_000)

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[:10_000]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[10_000:]

    def _pt_key(self, patient_id: int):
        return f"p{patient_id:05d}"

    @functools.cached_property
    def arr_rhythm_patients(self) -> npt.NDArray:
        """Find all patients with arrhythmia events. This takes roughly 10 secs.

        Returns:
            npt.NDArray: Patient ids

        """
        patient_ids = self.patient_ids.tolist()
        with Pool() as pool:
            arr_pts_bool = list(pool.imap(self._pt_has_rhythm_arrhythmia, patient_ids))
        patient_ids = np.where(arr_pts_bool)[0]
        return patient_ids

    def task_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Task-level data generator.

        Args:
            patient_generator (PatientGenerator): Patient data generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample data generator
        """
        if self.task in ["arrhythmia", "AFIB_Ident"]:
            return self.rhythm_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        if self.task == "noise_classify":
            return self.rhythm_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
                noise_filter=False,
            )
        if self.task == "beat":
            return self.beat_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        if self.task == "segmentation":
            return self.segmentation_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        raise NotImplementedError()

    def _split_train_test_patients(self, patient_ids: npt.NDArray, test_size: float) -> list[list[int]]:
        """Perform train/test split on patients for given task.
        NOTE: We only perform inter-patient splits and not intra-patient.

        Args:
            patient_ids (npt.NDArray): Patient Ids
            test_size (float): Test size

        Returns:
            list[list[int]]: Train and test sets of patient ids
        """
        # Use stratified split for arrhythmia task
        if self.task in ["arrhythmia", "AFIB_Ident", "noise_classify"]:
            arr_pt_ids = np.intersect1d(self.arr_rhythm_patients, patient_ids)
            norm_pt_ids = np.setdiff1d(patient_ids, arr_pt_ids)
            (
                norm_train_pt_ids,
                norm_val_pt_ids,
            ) = sklearn.model_selection.train_test_split(norm_pt_ids, test_size=test_size)
            (
                arr_train_pt_ids,
                afib_val_pt_ids,
            ) = sklearn.model_selection.train_test_split(arr_pt_ids, test_size=test_size)
            train_pt_ids = np.concatenate((norm_train_pt_ids, arr_train_pt_ids))
            val_pt_ids = np.concatenate((norm_val_pt_ids, afib_val_pt_ids))
            np.random.shuffle(train_pt_ids)
            np.random.shuffle(val_pt_ids)
            return train_pt_ids, val_pt_ids
        # END IF

        # Otherwise, use random split
        return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)

    def rhythm_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
        noise_filter: bool = True,
    ) -> SampleGenerator:
        """Generate frames w/ rhythm labels (e.g. afib) using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """
        # Target labels and mapping
        tgt_labels = list(set(self.class_map.values()))
        # Convert Icentia labels -> HK labels -> class map labels (-1 indicates not in class map)
        tgt_map = {k: self.class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
        num_classes = len(tgt_labels)

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            # now samples_per_tgt will match 
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_per_tgt * [num_classes]

        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))

        # Group patient rhythms by type (segment, start, stop, delta)
        for _, segments in patient_generator: # this for loop is traversing through all segments under every patient
            # This maps segment index to segment key
            seg_map: list[str] = list(segments.keys())

            pt_tgt_seg_map = [[] for _ in tgt_labels]
            for seg_idx, seg_key in enumerate(seg_map): # from s00~s49
                # Grab rhythm labels
                rlabels = segments[seg_key]["rlabels"][:]
                # we want to train on noise classification
                # Skip if no rhythm labels
                if not rlabels.shape[0]:
                    continue
                rlabels = rlabels[np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]]
                # Skip if only noise
                if not rlabels.shape[0]:
                    continue
                # Unpack start, end, and label
                xs, xe, xl = rlabels[0::2, 0], rlabels[1::2, 0], rlabels[0::2, 1]

                # Map labels to target labels
                xl = np.vectorize(tgt_map.get, otypes=[int])(xl)

                for tgt_idx, tgt_class in enumerate(tgt_labels): # this is extracting all the potential rlabels for every segment
                    idxs = np.where((xe - xs >= input_size) & (xl == tgt_class)) # we want to find a large enough frame which does contain the current class of label, we also make sure there will be no label mixing when preparing data
                    seg_vals = np.vstack((seg_idx * np.ones_like(idxs), xs[idxs], xe[idxs])).T
                    pt_tgt_seg_map[tgt_idx] += seg_vals.tolist()
                # END FOR
            # END FOR
            if not noise_filter:
                if any(len(b) > 0 for b in pt_tgt_seg_map):
                    pt_tgt_seg_map = np.concatenate([np.array(b) for b in pt_tgt_seg_map if len(b) > 0])
                else:
                    print("All arrays in pt_tgt_seg_map are empty.")
                    continue

                noise_seg = []
                # Iterate over the original array
                for i in range(len(pt_tgt_seg_map)-1):
                    # Append a new row to the new array
                    if pt_tgt_seg_map[i+1][1] - pt_tgt_seg_map[i][2] < input_size: # skip noise not long enough
                        continue
                    noise_seg.append([pt_tgt_seg_map[i][0], pt_tgt_seg_map[i][2], pt_tgt_seg_map[i+1][1]])
                noise_seg = np.array(noise_seg)
                pt_tgt_seg_map = [pt_tgt_seg_map, noise_seg]
                # tgt_labels
            else:    
                pt_tgt_seg_map = [np.array(b) for b in pt_tgt_seg_map] # pt_tgt_seg_map will be grouped by rlabel
                ## Sample Generation starts from here
                # Grab target segments
            seg_samples: list[tuple[int, int, int, int]] = []
            for tgt_idx, tgt_class in enumerate(tgt_labels):
                tgt_segments = pt_tgt_seg_map[tgt_idx]
                if not tgt_segments.shape[0]:
                    continue
                # print(f"Double check the tgt segments {tgt_segments}")
                tgt_seg_indices: list[int] = random.choices(
                    np.arange(tgt_segments.shape[0]),
                    weights=tgt_segments[:, 2] - tgt_segments[:, 1], # proportional to the length of every segment
                    k=samples_per_tgt[tgt_idx],
                ) # a number of indices equal to the number of samples for the current target label.

                # print(f"Overall tgt segment indices: {len(tgt_seg_indices)}") # should be 25 or 200 different tgt_seg_idx

                for tgt_seg_idx in tgt_seg_indices:
                    seg_idx, rhy_start, rhy_end = tgt_segments[tgt_seg_idx] # it should look something like this: [0     989   27860]
                    frame_start = np.random.randint(rhy_start, rhy_end - input_size + 1) # randomly select a frame within the segment sample
                    frame_end = frame_start + input_size
                    seg_samples.append((seg_idx, frame_start, frame_end, tgt_class))
                # END FOR
            # END FOR

            # Shuffle segments
            random.shuffle(seg_samples)

            # Yield selected samples for patient
            for seg_idx, frame_start, frame_end, label in seg_samples:
                x: npt.NDArray = segments[seg_map[seg_idx]]["data"][frame_start:frame_end].astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                yield x, label
            # END FOR
        # END FOR
                

    def segmentation_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Gnerate frames with annotated segments.

        Args:
            patient_generator (PatientGenerator): Patient generator
            samples_per_patient (int | list[int], optional):

        Returns:
            SampleGenerator: Sample generator
        """
        assert not isinstance(samples_per_patient, Iterable)
        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))

        # For each patient
        for _, segments in patient_generator:
            for _ in range(samples_per_patient):
                # Randomly pick a segment
                seg_key = np.random.choice(list(segments.keys()))
                # Randomly pick a frame
                frame_start = np.random.randint(segments[seg_key]["data"].shape[0] - input_size)
                frame_end = frame_start + input_size
                # Get data and labels
                data = segments[seg_key]["data"][frame_start:frame_end].squeeze()

                if self.sampling_rate != self.target_rate:
                    ds_ratio = self.target_rate / self.sampling_rate
                    data = pk.signal.resample_signal(data, self.sampling_rate, self.target_rate, axis=0)
                else:
                    ds_ratio = 1

                blabels = segments[seg_key]["blabels"]
                blabels = blabels[(blabels[:, 0] >= frame_start) & (blabels[:, 0] < frame_end)]
                # Create segment mask
                mask = np.zeros_like(data, dtype=np.int32)
                for i in range(blabels.shape[0]):
                    bidx = int((blabels[i, 0] - frame_start) * ds_ratio)
                    btype = blabels[i, 1]
                    if btype == IcentiaBeat.undefined:
                        continue

                    # Extract QRS segment
                    qrs = pk.signal.moving_gradient_filter(
                        data, sample_rate=self.target_rate, sig_window=0.1, avg_window=1.0, sig_prom_weight=1.5
                    )
                    win_len = max(1, int(0.08 * self.target_rate))  # 80 ms
                    b_left = max(0, bidx - win_len)
                    b_right = min(data.shape[0], bidx + win_len)
                    onset = np.where(np.flip(qrs[b_left:bidx]) < 0)[0]
                    onset = onset[0] if onset.size else win_len
                    offset = np.where(qrs[bidx + 1 : b_right] < 0)[0]
                    offset = offset[0] if offset.size else win_len
                    mask[bidx - onset : bidx + offset] = self.class_map.get(HeartSegment.qrs.value, 0)
                    # Ignore P, T, and U waves for now
                # END FOR
                x = np.nan_to_num(data).astype(np.float32)
                y = mask.astype(np.int32)
                yield x, y
            # END FOR
        # END FOR

    def beat_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames and beat label using patient generator.
        There are over 2.5 billion normal and undefined while less than 40 million arrhythmia beats.
        The following routine sorts each patient's beats by type and then approx. uniformly samples them by amount requested.

        Args:
            patient_generator (PatientGenerator): Patient generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """
        nlabel_threshold = 0.25
        blabel_padding = 20
        rr_win_len = int(10 * self.sampling_rate)
        rr_min_len = int(0.3 * self.sampling_rate)
        rr_max_len = int(2.0 * self.sampling_rate)

        # Target labels and mapping
        num_classes = len(set(self.class_map.values()))

        # Convert Icentia labels -> HK labels -> class map labels (-1 indicates not in class map)
        tgt_map = {k: self.class_map.get(v, -1) for (k, v) in HeartBeatMap.items()}

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_per_tgt * [num_classes]

        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))

        # For each patient
        for _, segments in patient_generator:
            # This maps segment index to segment key
            seg_map: list[str] = list(segments.keys())

            # Capture beat locations for each segment
            pt_beat_map = [[] for _ in range(num_classes)]
            for seg_idx, seg_key in enumerate(seg_map):
                # Get beat labels
                blabels = segments[seg_key]["blabels"][:]

                # If no beats, skip
                num_blabels = blabels.shape[0]
                if num_blabels <= 0:
                    continue
                # END IF

                # If too few normal beats, skip
                num_nlabels = np.sum(blabels[:, 1] == IcentiaBeat.normal)
                if num_nlabels / num_blabels < nlabel_threshold:
                    continue

                # Capture all beat locations
                # for tgt_beat_idx, beat in enumerate(tgt_labels):
                for beat in IcentiaBeat:
                    # Skip if not in class map
                    beat_class = tgt_map.get(beat, -1)
                    if beat_class < 0 or beat_class >= num_classes:
                        continue

                    # Get all beat type indices
                    beat_idxs = np.where(blabels[blabel_padding:-blabel_padding, 1] == beat.value)[0] + blabel_padding

                    # Filter indices based on beat type
                    if beat == IcentiaBeat.normal:
                        filt_func = lambda i: blabels[i - 1, 1] == blabels[i + 1, 1] == IcentiaBeat.normal
                    elif beat in (IcentiaBeat.pac, IcentiaBeat.pvc):
                        filt_func = lambda i: IcentiaBeat.undefined not in (
                            blabels[i - 1, 1],
                            blabels[i + 1, 1],
                        )
                    elif beat == IcentiaBeat.undefined:
                        filt_func = lambda i: blabels[i - 1, 1] == blabels[i + 1, 1] == IcentiaBeat.undefined
                    else:
                        filt_func = lambda _: True
                    # END IF

                    # Filter indices
                    beat_idxs = filter(filt_func, beat_idxs)
                    pt_beat_map[beat_class] += [(seg_idx, blabels[i, 0]) for i in beat_idxs]
                # END FOR
            # END FOR
            pt_beat_map = [np.array(b) for b in pt_beat_map]

            # Randomly select N samples of each target beat
            pt_segs_beat_idxs: list[tuple[int, int, int]] = []
            for tgt_beat_idx, tgt_beats in enumerate(pt_beat_map):
                tgt_count = min(samples_per_tgt[tgt_beat_idx], len(tgt_beats))
                tgt_idxs = np.random.choice(np.arange(len(tgt_beats)), size=tgt_count, replace=False)
                pt_segs_beat_idxs += [(tgt_beats[i][0], tgt_beats[i][1], tgt_beat_idx) for i in tgt_idxs]
            # END FOR

            # Shuffle all
            random.shuffle(pt_segs_beat_idxs)

            # Yield selected samples for patient
            for seg_idx, beat_idx, beat in pt_segs_beat_idxs:
                frame_start = max(0, beat_idx - int(random.uniform(0.4722, 0.5278) * input_size))
                frame_end = frame_start + input_size
                data = segments[seg_map[seg_idx]]["data"]
                blabels = segments[seg_map[seg_idx]]["blabels"]

                # Compute average RR interval
                rr_xs = np.searchsorted(blabels[:, 0], max(0, frame_start - rr_win_len))
                rr_xe = np.searchsorted(blabels[:, 0], frame_end + rr_win_len)
                if rr_xe <= rr_xs:
                    continue
                rri = np.diff(blabels[rr_xs : rr_xe + 1, 0])
                rri = rri[(rri > rr_min_len) & (rri < rr_max_len)]
                if rri.size <= 0:
                    continue
                avg_rr = int(np.mean(rri))

                if frame_start - avg_rr < 0 or frame_end + avg_rr >= data.shape[0]:
                    continue

                # Combine previous, current, and next beat
                x = np.hstack(
                    (
                        data[frame_start - avg_rr : frame_end - avg_rr],
                        data[frame_start:frame_end],
                        data[frame_start + avg_rr : frame_end + avg_rr],
                    )
                )
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                y = beat
                yield x, y
            # END FOR
        # END FOR

    def heart_rate_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int = 1,
    ) -> SampleGenerator:
        """Generate frames and heart rate label using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient generator
            samples_per_patient (int, optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """

        label_frame_size = self.frame_size
        max_frame_size = max(self.frame_size, label_frame_size)
        for _, segments in patient_generator:
            for _ in range(samples_per_patient):
                segment = segments[np.random.choice(list(segments.keys()))]
                segment_size: int = segment["data"].shape[0]
                frame_center = np.random.randint(segment_size - max_frame_size) + max_frame_size // 2
                signal_frame_start = frame_center - self.frame_size // 2
                signal_frame_end = frame_center + self.frame_size // 2
                x = segment["data"][signal_frame_start:signal_frame_end]
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                label_frame_start = frame_center - label_frame_size // 2
                label_frame_end = frame_center + label_frame_size // 2
                beat_indices = segment["blabels"][:, 0]
                frame_beat_indices = self.get_complete_beats(beat_indices, start=label_frame_start, end=label_frame_end)
                y = self._get_heart_rate_label(frame_beat_indices, self.sampling_rate)
                yield x, y
            # END FOR
        # END FOR

    def signal_generator(self, patient_generator: PatientGenerator, samples_per_patient: int = 1) -> SampleGenerator:
        """Generate random frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))
        for _, segments in patient_generator:
            for _ in range(samples_per_patient):
                segment = segments[np.random.choice(list(segments.keys()))]
                segment_size = segment["data"].shape[0]
                frame_start = np.random.randint(segment_size - input_size)
                frame_end = frame_start + input_size
                x = segment["data"][frame_start:frame_end].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                # END IF
                yield x
            # END FOR
        # END FOR
    
    def signal_label_generator(self, patient_generator: PatientGenerator, samples_per_patient: int = 1) -> SampleGenerator:
        """Generate random frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient. should be 1 in this case

        Returns:
            SampleGenerator: Generator of signal and their corresponding label (frame_size, 1)
        """
        # tgt_map = {k: self.class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
        # class_map = get_class_mapping(self.num_classes)
        # tgt_map = {k: class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))
        for _, segments in patient_generator:
            for _ in range(samples_per_patient): # we will run it 25 times or 200 times
                segment_id = np.random.choice(list(segments.keys())) # randomly pick a segment
                segment = segments[segment_id]
                # get the overall size of current segment, _sID
                segment_size = segment["data"].shape[0]
                # get all the peak rlabels for current segment
                rlabels = segment["rlabels"][:]

                # xs, xe, xl = rlabels[0::2, 0], rlabels[1::2, 0], rlabels[0::2, 1]
                # randomly select a frame start
                frame_start = np.random.randint(segment_size - input_size)
                frame_end = frame_start + input_size
                # no rlabel
                if len(rlabels) == 0:
                    print("Current sample contains no rlables skip it for now")
                    continue
                else:
                    # rlabels = segments[seg_key]["rlabels"][:]
                    # Unpack start, end, and label
                    # xs, xe, xl = rlabels[0::2, 0], rlabels[1::2, 0], rlabels[0::2, 1]
                    # Get the first elements of each sub-array
                    sig_pos = rlabels[:, 0]
                    # print(f"Double check {rlabel}")
                    in_range = np.logical_and(frame_start <= sig_pos, frame_end >= sig_pos)
                    # Get the indices of the elements that are within the range
                    indices = np.where(in_range)[0]
                    # current frame does not cover the peak
                    if len(indices) == 0:
                        # current frame does not contain any peak, get the closet two intervals
                        # frame_rlabel = np.array([rlabel[sig_pos <= frame_end][-1], rlabel[frame_start <= sig_pos][0]])
                        array1 = rlabels[sig_pos <= frame_end]
                        array2 = rlabels[frame_start <= sig_pos]
                        # Check if both arrays are non-empty
                        if len(array1) > 0 and len(array2) > 0:
                            # If both are non-empty, get the last element of array1 and the first element of array2
                            frame_rlabel = np.array([array1[-1], array2[0]])
                        elif len(array1) > 0:
                            # If only array1 is non-empty, get its last element
                            frame_rlabel = np.array([array1[-1]])
                        elif len(array2) > 0:
                            # If only array2 is non-empty, get its first element
                            frame_rlabel = np.array([array2[0]])
                        else:
                            # If both are empty, set frame_rlabel as an empty array
                            print("Cannot find the sample in current segment")
                            frame_rlabel = np.full(segment_size, -1)
                        # if frame_rlabel not in [2,3]:
                    else:
                        frame_rlabel = rlabels[indices]
                    # END if
                    frame_rlabel[:, 0] -= frame_start

                x = segment["data"][frame_start:frame_end].squeeze() # this step we are also removing all the position indices for x, x will always be [0, input_size]
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                
                # END IF
                yield x, frame_rlabel, segment_id
            # END FOR
        # END FOR
                
    def signal_label_TimeFrame_generator(self, patient_generator: PatientGenerator, segment_idx, frame_start, frame_end) -> SampleGenerator:
        """Generate random frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient. should be 1 in this case

        Returns:
            SampleGenerator: Generator of signal and their corresponding label (frame_size, 1)
        """
        for _, segments in patient_generator:
            segment_id = list(segments.keys())[segment_idx]
            segment = segments[segment_id]
            # get the overall size of current segment, _sID
            # segment_size = segment["data"].shape[0]
            # get all the peak rlabels for current segment
            rlabels = segment["rlabels"][:]
            # print(rlabels)
            # no rlabel
            if len(rlabels) == 0:
                print("Current sample contains no rlables skip it for now")
                continue
            else:
                if rlabels[:, 0][-1] < frame_end:
                    print("Current sample is larger than the maximal frame size")
                    continue
                # an rlabel array for the whole segment
                whole_seg_rlabels = np.zeros(rlabels[:, 0][-1])
                # print(f"Length of current frame rlabel: {rlabels[:, 0][-1]}")
                # pos_start = rlabels[:, 0][rlabels[:, 1] != 4]
                pos_start = rlabels[:, 0][(rlabels[:, 1] != 0) & (rlabels[:, 1] != 4)]
                pos_end = rlabels[:, 0][rlabels[:, 1] == 4]
                pos_idx = rlabels[:, 1][rlabels[:, 1] != 4]
                pos_ranges = np.vstack((pos_start, pos_end)).T
                # print(f"Frame ranges: {pos_ranges}, {frame_start}, {frame_end}")
                for i in range(len(pos_ranges)):
                    start, end = pos_ranges[i]
                    whole_seg_rlabels[start:end] = pos_idx[i]
                frame_rlabel = whole_seg_rlabels[frame_start:frame_end]
                # print(f"Double check {frame_rlabel}")
            x = segment["data"][frame_start:frame_end].squeeze() # this step we are also removing all the position indices for x, x will always be [0, input_size]
            x = np.nan_to_num(x).astype(np.float32)
           #  print(f"Before Resampling the signal {len(x)}")
            if self.sampling_rate != self.target_rate:
                x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                # print(f"Resample the signal {len(x)}")
            # END IF
            yield x, frame_rlabel, segment_id
            # END FOR
        # END FOR

    # TODO: upgrade the main inferen driver
    def signal_label_segment_generator(self, patient_generator: PatientGenerator, segment_idx, timerange=range(60)) -> SampleGenerator:
        """Generate random frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient. should be 1 in this case

        Returns:
            SampleGenerator: Generator of signal and their corresponding label (frame_size, 1)
        """
        for _, segments in patient_generator:
            segment_id = list(segments.keys())[segment_idx]
            segment = segments[segment_id]
            rlabels = segment["rlabels"][:]
            if len(rlabels) == 0:
                print("Current sample contains no rlables skip it for now")
                continue
            else:
                frame_start = timerange[0]
                frame_end = timerange[1]
                # print(f"frame ends here: {self.target_rate}, {timerange[-1]+1}")
                if rlabels[:, 0][-1] < frame_end:
                    print("Current sample is larger than the maximal frame size")
                    continue
                # an rlabel array for the whole segment
                whole_seg_rlabels = np.zeros(rlabels[:, 0][-1])
                pos_start = rlabels[:, 0][(rlabels[:, 1] != 0) & (rlabels[:, 1] != 4)]
                pos_end = rlabels[:, 0][rlabels[:, 1] == 4]
                pos_idx = rlabels[:, 1][rlabels[:, 1] != 4]
                pos_ranges = np.vstack((pos_start, pos_end)).T
                for i in range(len(pos_ranges)):
                    start, end = pos_ranges[i]
                    whole_seg_rlabels[start:end] = pos_idx[i]
                
                
                frame_rlabel = whole_seg_rlabels[frame_start:frame_end]
                # print(f"Double check {frame_rlabel}")
            x = segment["data"][frame_start:frame_end].squeeze() # this step we are also removing all the position indices for x, x will always be [0, input_size]
            x = np.nan_to_num(x).astype(np.float32)
            # print(len(frame_rlabel), len(x), self.sampling_rate, self.target_rate)
            if self.sampling_rate != self.target_rate:
                x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                # print(f"Resample the signal {len(x)}")
            
            # END IF
            yield x, frame_rlabel
            # END FOR
        # END FOR
            
    def signal_rblabel_generator(self, patient_generator: PatientGenerator, segment_idx="all", timerange=None) -> SampleGenerator:
        """Generate random frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient. should be 1 in this case

        Returns:
            SampleGenerator: Generator of signal and their corresponding label (frame_size, 1)
        """
        for _, segments in patient_generator:
            segment_id = list(segments.keys())[segment_idx]
            segment = segments[segment_id]
            # get all the peak rlabels for current segment
            blabels = segment["blabels"][:]
            # print(blabels)
            if len(blabels) <= 0:
                print("This segment has no beat label")
                continue
            rlabels = segment["rlabels"][:]
            # no rlabel
            if len(rlabels) == 0:
                print("Current sample contains no rlables skip it for now")
                continue
            else:
                if timerange is not None:
                    frame_start = timerange[0]
                    frame_end = timerange[1]
                else:
                    frame_blabel = blabels[:, 0][blabels[:, 1] == 1]
                    # print(frame_blabel)
                    frame_start = rlabels[:, 0][0]
                    frame_end = rlabels[:, 0][-1]
                # print(f"frame ends here: {self.target_rate}, {timerange[-1]+1}")
                if rlabels[:, 0][-1] < frame_end:
                    print("Current sample is larger than the maximal frame size")
                    continue
                # rlabel extraction
                whole_seg_rlabels = np.zeros(rlabels[:, 0][-1])
                pos_start = rlabels[:, 0][(rlabels[:, 1] != 0) & (rlabels[:, 1] != 4)]
                pos_end = rlabels[:, 0][rlabels[:, 1] == 4]
                pos_idx = rlabels[:, 1][rlabels[:, 1] != 4]
                pos_ranges = np.vstack((pos_start, pos_end)).T
                for i in range(len(pos_ranges)):
                    start, end = pos_ranges[i]
                    whole_seg_rlabels[start:end] = pos_idx[i]
                frame_rlabel = whole_seg_rlabels[frame_start:frame_end]
                if frame_end != rlabels[:, 0][-1]:
                    # blabel extraction
                    whole_seg_blabels = np.zeros(blabels[:, 0][-1])
                    pos_idx = blabels[:, 0][blabels[:, 1] == 1]
                    for i in range(len(pos_idx)-1):
                        whole_seg_blabels[pos_idx[i]] = 1
                    frame_blabel = whole_seg_blabels[frame_start:frame_end]

            x = segment["data"][frame_start:frame_end].squeeze() # this step we are also removing all the position indices for x, x will always be [0, input_size]
            x = np.nan_to_num(x).astype(np.float32)
            if self.sampling_rate != self.target_rate:
                x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
            # print(x, frame_rlabel, frame_blabel)
            # END IF
            yield x, frame_rlabel, frame_blabel
            # END FOR
        # END FOR
                

    def continous_signal_label_generator(self, patient_generator: PatientGenerator, selected_time: int = 15) -> SampleGenerator:
        """Generate continous frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            selected_time (int): Number of minutes per patient

        Returns:
            SampleGenerator: Generator of signal and their corresponding label (frame_size, 1)
        """
        # tgt_map = {k: self.class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
        # class_map = get_class_mapping(self.num_classes)
        # tgt_map = {k: class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
        time_duration = [1, 15, 60, 240, 1440] # the key concept here the data we extract is no longer randomly sampled, they are continou for that long duration
        if selected_time not in time_duration:
            raise KeyError(f"Selected time {selected_time} is not in the list of allowed durations: {time_duration}")
        seg_req = selected_time // 60
        
        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size)) # 4 seconds sample frame size
        time_frame = input_size * (60 * self.target_rate/self.frame_size) * 60 if selected_time >= 60 else selected_time # 60 * 100/400
        for _, segments in patient_generator:
            seg_req = selected_time // 60
            segment_ids = np.random.choice(list(segments.keys()), size=1 if selected_time < 60 else seg_req, replace=False)
            merged_seg = []
            for segment_id in segment_ids:
                segment = segments[segment_id]
                # get the overall size of current segment, _sID, should be approximately 70 minutes
                segment_size = segment["data"].shape[0]
                # get all the peak rlabels for current segment
                rlabels = segment["rlabels"][:]
                frame_start = np.random.randint(segment_size - time_frame)
                frame_end = int(frame_start + time_frame)
                print(f"Starting idx is: {frame_start}, and end idx is: {frame_end}")
                # no rlabel
                if len(rlabels) == 0:
                    print("Current sample contains no rlables skip it for now")
                    continue
                else:
                    sig_pos = rlabels[:, 0]
                    # print(f"Double check {rlabel}")
                    in_range = np.logical_and(frame_start <= sig_pos, frame_end >= sig_pos)
                    # Get the indices of the elements that are within the range
                    indices = np.where(in_range)[0]
                    # current frame does not cover the peak
                    if len(indices) == 0:
                        array1 = rlabels[sig_pos <= frame_end]
                        array2 = rlabels[frame_start <= sig_pos]
                        # Check if both arrays are non-empty
                        if len(array1) > 0 and len(array2) > 0:
                            # If both are non-empty, get the last element of array1 and the first element of array2
                            frame_rlabel = np.array([array1[-1], array2[0]])
                        elif len(array1) > 0:
                            # If only array1 is non-empty, get its last element
                            frame_rlabel = np.array([array1[-1]])
                        elif len(array2) > 0:
                            # If only array2 is non-empty, get its first element
                            frame_rlabel = np.array([array2[0]])
                        else:
                            # If both are empty, set frame_rlabel as an empty array
                            print("Cannot find the sample in current segment")
                            frame_rlabel = np.full(segment_size, -1)
                        # if frame_rlabel not in [2,3]:
                    else:
                        frame_rlabel = rlabels[indices]
                    # END if
                    frame_rlabel[:, 0] -= frame_start

                x = segment["data"][frame_start:frame_end].squeeze() # this step we are also removing all the position indices for x, x will always be [0, input_size]
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                # END IF
                # Append the results to the list as a tuple
                merged_seg.append({'x': x, 'frame_rlabel': frame_rlabel, 'segment_id': segment_id})

            # Convert the list of dictionaries to a pandas DataFrame
            # merged_seg = pd.DataFrame(all_segment_results)
            yield merged_seg
        


    def uniform_patient_generator(
        self,
        patient_ids: npt.NDArray,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> PatientGenerator:
        """Yield data for each patient in the array.

        Args:
            patient_ids (pt.ArrayLike): Array of patient ids
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle patient ids.. Defaults to True.

        Returns:
            PatientGenerator: Patient generator

        Yields:
            Iterator[PatientGenerator]
        """
        patient_ids = np.copy(patient_ids)
        while True:
            if shuffle:
                np.random.shuffle(patient_ids)
            for patient_id in patient_ids:
                pt_key = self._pt_key(patient_id)
                with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
                    patient_data = h5[pt_key]
                    yield patient_id, patient_data
                # END WITH
            # END FOR
            if not repeat:
                break
            # END IF
        # END WHILE

    def random_patient_generator(
        self,
        patient_ids: list[int],
        patient_weights: list[int] | None = None,
    ) -> PatientGenerator:
        """Samples patient data from the provided patient distribution.

        Args:
            patient_ids (list[int]): Patient ids
            patient_weights (list[int] | None, optional): Probabilities associated with each patient. Defaults to None.

        Returns:
            PatientGenerator: Patient generator

        Yields:
            Iterator[PatientGenerator]
        """
        while True:
            for patient_id in np.random.choice(patient_ids, size=1024, p=patient_weights):
                pt_key = self._pt_key(patient_id)
                with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
                    patient_data = h5[pt_key]
                    yield patient_id, patient_data
                # END WITH
            # END FOR
        # END WHILE

    def get_complete_beats(
        self,
        indices: npt.NDArray,
        labels: npt.NDArray | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Find all complete beats within a frame i.e. start and end of the beat lie within the frame.
        The indices are assumed to specify the end of a heartbeat.

        Args:
            indices (npt.NDArray): List of sorted beat indices.
            labels (npt.NDArray | None): List of beat labels. Defaults to None.
            start (int): Index of the first sample in the frame. Defaults to 0.
            end (int | None): Index of the last sample in the frame. Defaults to None.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: (beat indices, beat labels)
        """
        if end is None:
            end = indices[-1]
        if start >= end:
            raise ValueError("`end` must be greater than `start`")
        start_index = np.searchsorted(indices, start, side="left") + 1
        end_index = np.searchsorted(indices, end, side="right")
        indices_slice = indices[start_index:end_index]
        if labels is None:
            return indices_slice
        label_slice = labels[start_index:end_index]
        return (indices_slice, label_slice)

    def _get_heart_rate_label(self, qrs_indices, fs=None) -> int:
        """Determine the heart rate label based on an array of QRS indices (separating individual heartbeats).
            The QRS indices are assumed to be measured in seconds if sampling frequency `fs` is not specified.
            The heartbeat label is based on the following BPM (beats per minute) values: (0) tachycardia <60 BPM,
            (1) bradycardia >100 BPM, (2) healthy 60-100 BPM, (3) noisy if QRS detection failed.

        Args:
            qrs_indices (list[int]): Array of QRS indices.
            fs (float, optional): Sampling frequency of the signal. Defaults to None.

        Returns:
            int: Heart rate label
        """
        if not qrs_indices:
            return HeartRate.noise.value

        rr_intervals = np.diff(qrs_indices)
        if fs is not None:
            rr_intervals = rr_intervals / fs
        bpm = 60 / rr_intervals.mean()
        if bpm < 60:
            return HeartRate.bradycardia.value
        if bpm <= 100:
            return HeartRate.normal.value
        return HeartRate.tachycardia.value

    def _pt_has_rhythm_arrhythmia(self, patient_id: int):
        pt_key = self._pt_key(patient_id)
        with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
            for _, segment in h5[pt_key].items():
                rlabels = segment["rlabels"][:]
                if not rlabels.shape[0]:
                    continue
                rlabels = rlabels[:, 1]
                if len(np.where((rlabels == IcentiaRhythm.afib) | (rlabels == IcentiaRhythm.aflut))[0]):
                    return True
            return False
    
    def get_pseg_ref(self, patient_generator: PatientGenerator, segment_idx) -> SampleGenerator:
        """Generate random frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient. should be 1 in this case

        Returns:
            SampleGenerator: Generator of signal and their corresponding label (frame_size, 1)
        """
        for _, segments in patient_generator:
            segment_id = list(segments.keys())[segment_idx]
            segment = segments[segment_id]
            # get the overall size of current segment, _sID
            # segment_size = segment["data"].shape[0]
            # get all the peak rlabels for current segment
            rlabels = segment["rlabels"][:]
            # no rlabel
            if len(rlabels) == 0:
                print("Current sample contains no rlables skip it for now")
                continue
            else:
                # an rlabel array for the whole segment
                whole_seg_rlabels = np.zeros(rlabels[:, 0][-1])
                pos_start = rlabels[:, 0][rlabels[:, 1] != 4]
                pos_end = rlabels[:, 0][rlabels[:, 1] == 4]
                pos_idx = rlabels[:, 1][rlabels[:, 1] != 4]
                pos_ranges = np.vstack((pos_start, pos_end)).T
                for i in range(len(pos_ranges)):
                    start, end = pos_ranges[i]
                    whole_seg_rlabels[start:end] = pos_idx[i]
            yield whole_seg_rlabels

    def get_rhythm_statistics(
        self,
        patient_ids: npt.NDArray | None = None,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """Utility function to extract rhythm statistics across entire dataset. Useful for EDA.

        Args:
            patient_ids (npt.NDArray | None, optional): Patients IDs to include. Defaults to all.
            save_path (str | None, optional): Parquet file path to save results. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of statistics
        """

        if patient_ids is None:
            patient_ids = self.patient_ids
        pt_gen = self.uniform_patient_generator(patient_ids=patient_ids, repeat=False)
        stats = []
        for pt, segments in pt_gen:
            # Group patient rhythms by type (segment, start, stop)
            segment_label_map: dict[str, list[tuple[str, int, int]]] = {}
            for seg_key, segment in segments.items():
                rlabels = segment["rlabels"][:]
                if rlabels.shape[0] == 0:
                    continue  # Segment has no rhythm labels
                rlabels = rlabels[np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]]
                for i, l in enumerate(rlabels[::2, 1]):
                    if l in (
                        IcentiaRhythm.noise,
                        IcentiaRhythm.normal,
                        IcentiaRhythm.afib,
                        IcentiaRhythm.aflut,
                    ):
                        rhy_start, rhy_stop = (
                            rlabels[i * 2 + 0, 0],
                            rlabels[i * 2 + 1, 0],
                        )
                        stats.append(
                            dict(
                                pt=pt,
                                rc=seg_key,
                                rhythm=l,
                                start=rhy_start,
                                stop=rhy_stop,
                                dur=rhy_stop - rhy_start,
                            )
                        )
                        segment_label_map[l] = segment_label_map.get(l, []) + [
                            (seg_key, rlabels[i * 2 + 0, 0], rlabels[i * 2 + 1, 0])
                        ]
                    # END IF
                # END FOR
            # END FOR
        # END FOR
        df = pd.DataFrame(stats)
        if save_path:
            df.to_parquet(save_path)
        return df
    
    def get_noise_rhythm_statistics(
        self,
        patient_ids: npt.NDArray | None = None,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """Utility function to extract rhythm statistics across entire dataset. Useful for EDA.

        Args:
            patient_ids (npt.NDArray | None, optional): Patients IDs to include. Defaults to all.
            save_path (str | None, optional): Parquet file path to save results. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of statistics
        """

        if patient_ids is None:
            patient_ids = self.patient_ids
        pt_gen = self.uniform_patient_generator(patient_ids=patient_ids, repeat=False)
        stats = []
        for pt, segments in pt_gen:
            # Group patient rhythms by type (segment, start, stop)
            segment_label_map: dict[str, list[tuple[str, int, int]]] = {}
            for seg_key, segment in segments.items():
                rlabels = segment["rlabels"][:]
                if rlabels.shape[0] == 0:
                    continue  # Segment has no rhythm labels
                rlabels = rlabels[np.where(rlabels[:, 1] == IcentiaRhythm.noise.value)[0]]
                for i, l in enumerate(rlabels[::2, 1]):
                    if l in (
                        IcentiaRhythm.noise,
                        IcentiaRhythm.normal,
                        IcentiaRhythm.afib,
                        IcentiaRhythm.aflut,
                    ):
                        rhy_start, rhy_stop = (
                            rlabels[i * 2 + 0, 0],
                            rlabels[i * 2 + 1, 0],
                        )
                        stats.append(
                            dict(
                                pt=pt,
                                rc=seg_key,
                                rhythm=l,
                                start=rhy_start,
                                stop=rhy_stop,
                                dur=rhy_stop - rhy_start,
                            )
                        )
                        segment_label_map[l] = segment_label_map.get(l, []) + [
                            (seg_key, rlabels[i * 2 + 0, 0], rlabels[i * 2 + 1, 0])
                        ]
                    # END IF
                # END FOR
            # END FOR
        # END FOR
        df = pd.DataFrame(stats)
        if save_path:
            df.to_parquet(save_path)
        return df
    
    def get_abnorm_rhythm_statistics(
        self,
        patient_ids: npt.NDArray | None = None,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """Utility function to extract beat statistics across entire dataset. Useful for EDA.

        Args:
            patient_ids (npt.NDArray | None, optional): Patients IDs to include. Defaults to all.
            save_path (str | None, optional): Parquet file path to save results. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of statistics
        """

        if patient_ids is None:
            patient_ids = self.patient_ids
        pt_gen = self.uniform_patient_generator(patient_ids=patient_ids, repeat=False)
        stats = []
        for pt, segments in pt_gen:
            # Group patient rhythms by type (segment, start, stop)
            segment_label_map: dict[str, list[tuple[str, int, int]]] = {}
            for seg_key, segment in segments.items():
                rlabels = segment["rlabels"][:]
                blabels = segment["blabels"][:]
                if rlabels.shape[0] == 0:
                    continue  # Segment has no rhythm labels
                # rlabels = rlabels[np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]]
                for i, l in enumerate(rlabels[::2, 1]):
                    if l in (
                        IcentiaRhythm.noise,
                        IcentiaRhythm.normal,
                        IcentiaRhythm.afib,
                        IcentiaRhythm.aflut,
                    ):
                        if (i * 2 + 1) >= len(rlabels):
                            break
                        rhy_start, rhy_stop = (
                            rlabels[i * 2 + 0, 0],
                            rlabels[i * 2 + 1, 0],
                        )
                        cur_beat = blabels[rhy_start:rhy_stop, ]
                        abnorm_beat = cur_beat[np.where(cur_beat[:, 1] == IcentiaBeat.undefined.value)[0]]

                        stats.append(
                            dict(
                                pt=pt,
                                rc=seg_key,
                                rhythm=l,
                                start=rhy_start,
                                stop=rhy_stop,
                                dur=rhy_stop - rhy_start,
                                noise_beat=abnorm_beat[:, 0],
                            )
                        )
                        segment_label_map[l] = segment_label_map.get(l, []) + [
                            (seg_key, rlabels[i * 2 + 0, 0], rlabels[i * 2 + 1, 0])
                        ]
                    # END IF
                # END FOR
            # END FOR
        # END FOR
        df = pd.DataFrame(stats)
        if save_path:
            df.to_parquet(save_path)
        return df


    def get_noise_beat_statistics(
        self,
        patient_ids: npt.NDArray | None = None,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """Utility function to extract beat statistics across entire dataset. Useful for EDA.

        Args:
            patient_ids (npt.NDArray | None, optional): Patients IDs to include. Defaults to all.
            save_path (str | None, optional): Parquet file path to save results. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of abnormal statistics
        """

        if patient_ids is None:
            patient_ids = self.patient_ids
        pt_gen = self.uniform_patient_generator(patient_ids=patient_ids, repeat=False)
        stats = []
        for pt, segments in pt_gen:
            # Group patient rhythms by type (segment, start, stop)
            segment_label_map: dict[str, list[tuple[str, int, int]]] = {}
            for seg_key, segment in segments.items():
                blabels = segment["blabels"][:]
                if blabels.shape[0] == 0:
                    continue  # Segment has no beat labels

                # filter undefined peaks
                blabels = blabels[np.where(blabels[:, 1] == IcentiaBeat.undefined.value)[0]]

                # print(len(blabels[::2, 1]), len(blabels[::2, 0]), len(blabels))
                for i, l in enumerate(blabels[::2, 1]):
                    if l in (
                        IcentiaBeat.undefined,
                        IcentiaBeat.normal,
                        IcentiaBeat.pac,
                        IcentiaBeat.aberrated,
                        IcentiaBeat.pvc,
                    ):  
                        if (i * 2 + 1) >= len(blabels):
                            break
                        beat_start, beat_stop = (
                            blabels[i * 2 + 0, 0],
                            blabels[i * 2 + 1, 0],
                        )
                        stats.append(
                            dict(
                                pt=pt,
                                rc=seg_key,
                                rhythm=l,
                                start=beat_start,
                                stop=beat_stop,
                                dur=beat_stop - beat_start,
                            )
                        )
                        segment_label_map[l] = segment_label_map.get(l, []) + [
                            (seg_key, blabels[i * 2 + 0, 0], blabels[i * 2 + 1, 0])
                        ]
                    # END IF
                # END FOR
            # END FOR
        # END FOR
        df = pd.DataFrame(stats)
        if save_path:
            df.to_parquet(save_path)
        return df
    

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        This will download preprocessed HDF5 files from S3.

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """

        def download_s3_file(
            s3_file: str,
            save_path: os.PathLike,
            bucket: str,
            client: boto3.client,
            force: bool = False,
        ):
            if not force and os.path.exists(save_path):
                return
            client.download_file(
                Bucket=bucket,
                Key=s3_file,
                Filename=str(save_path),
            )

        s3_bucket = "ambiqai-ecg-icentia11k-dataset"
        s3_prefix = "patients"

        os.makedirs(self.ds_path, exist_ok=True)

        patient_ids = self.patient_ids

        # Creating only one session and one client
        session = boto3.Session()
        client = session.client("s3", config=Config(signature_version=UNSIGNED))

        func = functools.partial(download_s3_file, bucket=s3_bucket, client=client, force=force)

        with tqdm(desc="Downloading icentia11k dataset from S3", total=len(patient_ids)) as pbar:
            pt_keys = [self._pt_key(patient_id) for patient_id in patient_ids]
            with ThreadPoolExecutor(max_workers=2 * num_workers) as executor:
                futures = (
                    executor.submit(
                        func,
                        f"{s3_prefix}/{pt_key}.h5",
                        self.ds_path / f"{pt_key}.h5",
                    )
                    for pt_key in pt_keys
                )
                for future in as_completed(futures):
                    err = future.exception()
                    if err:
                        logger.exception("Failed on file")
                    pbar.update(1)
                # END FOR
            # END WITH
        # END WITH

    def download_raw_dataset(self, num_workers: int | None = None, force: bool = False):
        """Downloads full Icentia dataset zipfile and converts into individial patient HDF5 files.
        NOTE: This is a very long process (e.g. 24 hrs). Please use `icentia11k.download_dataset` instead.

        Args:
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        logger.info("Downloading icentia11k dataset")
        ds_url = (
            "https://physionet.org/static/published-projects/icentia11k-continuous-ecg/"
            "icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip"
        )
        ds_zip_path = self.ds_path / "icentia11k.zip"
        os.makedirs(self.ds_path, exist_ok=True)
        if os.path.exists(ds_zip_path) and not force:
            logger.warning(
                f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={ds_zip_path}"
            )
        else:
            download_file(ds_url, ds_zip_path, progress=True)

        # 2. Extract and convert patient ECG data to H5 files
        logger.info("Generating icentia11k patient data")
        self._convert_dataset_zip_to_hdf5(
            zip_path=ds_zip_path,
            force=force,
            num_workers=num_workers,
        )
        logger.info("Finished icentia11k patient data")

    def _convert_dataset_pt_zip_to_hdf5(self, patient: int, zip_path: os.PathLike, force: bool = False):
        """Extract patient data from Icentia zipfile. Pulls out ECG data along with all labels.

        Args:
            patient (int): Patient id
            zip_path (PathLike): Zipfile path
            force (bool, optional): Whether to override destination if it exists. Defaults to False.
        """
        import re  # pylint: disable=import-outside-toplevel

        import wfdb  # pylint: disable=import-outside-toplevel

        # These map Wfdb labels to icentia labels
        WfdbRhythmMap = {
            "": IcentiaRhythm.noise.value,
            "(N": IcentiaRhythm.normal.value,
            "(AFIB": IcentiaRhythm.afib.value,
            "(AFL": IcentiaRhythm.aflut.value,
            ")": IcentiaRhythm.end.value,
        }
        WfdbBeatMap = {
            "Q": IcentiaBeat.undefined.value,
            "N": IcentiaBeat.normal.value,
            "S": IcentiaBeat.pac.value,
            "a": IcentiaBeat.aberrated.value,
            "V": IcentiaBeat.pvc.value,
        }

        logger.info(f"Processing patient {patient}")
        pt_id = self._pt_key(patient)
        pt_path = self.ds_path / f"{pt_id}.h5"
        if not force and os.path.exists(pt_path):
            logger.debug(f"Skipping patient {pt_id}")
            return
        zp = zipfile.ZipFile(zip_path, mode="r")  # pylint: disable=consider-using-with
        h5 = h5py.File(pt_path, mode="w")

        # Find all patient .dat file indices
        zp_rec_names = filter(
            lambda f: re.match(f"{pt_id}_[A-z0-9]+.dat", os.path.basename(f)),
            (f.filename for f in zp.filelist),
        )
        for zp_rec_name in zp_rec_names:
            try:
                zp_hdr_name = zp_rec_name.replace(".dat", ".hea")
                zp_atr_name = zp_rec_name.replace(".dat", ".atr")

                with tempfile.TemporaryDirectory() as tmpdir:
                    rec_fpath = os.path.join(tmpdir, os.path.basename(zp_rec_name))
                    atr_fpath = rec_fpath.replace(".dat", ".atr")
                    hdr_fpath = rec_fpath.replace(".dat", ".hea")
                    with open(hdr_fpath, "wb") as fp:
                        fp.write(zp.read(zp_hdr_name))
                    with open(rec_fpath, "wb") as fp:
                        fp.write(zp.read(zp_rec_name))
                    with open(atr_fpath, "wb") as fp:
                        fp.write(zp.read(zp_atr_name))
                    rec = wfdb.rdrecord(os.path.splitext(rec_fpath)[0], physical=True)
                    atr = wfdb.rdann(os.path.splitext(atr_fpath)[0], extension="atr")
                pt_seg_path = f"/{os.path.splitext(os.path.basename(zp_rec_name))[0].replace('_', '/')}"
                data = rec.p_signal.astype(np.float16)
                blabels = np.array(
                    [[atr.sample[i], WfdbBeatMap.get(s)] for i, s in enumerate(atr.symbol) if s in WfdbBeatMap],
                    dtype=np.int32,
                )
                rlabels = np.array(
                    [
                        [atr.sample[i], WfdbRhythmMap.get(atr.aux_note[i], 0)]
                        for i, s in enumerate(atr.symbol)
                        if s == "+"
                    ],
                    dtype=np.int32,
                )
                h5.create_dataset(
                    name=f"{pt_seg_path}/data",
                    data=data,
                    compression="gzip",
                    compression_opts=3,
                )
                h5.create_dataset(name=f"{pt_seg_path}/blabels", data=blabels)
                h5.create_dataset(name=f"{pt_seg_path}/rlabels", data=rlabels)
            except Exception as err:  # pylint: disable=broad-except
                logger.warning(f"Failed processing {zp_rec_name}", err)
                continue
        h5.close()

    def _convert_dataset_zip_to_hdf5(
        self,
        zip_path: os.PathLike,
        patient_ids: npt.NDArray | None = None,
        force: bool = False,
        num_workers: int | None = None,
    ):
        """Convert zipped Icentia dataset into individial patient HDF5 files.

        Args:
            zip_path (PathLike): Zipfile path
            patient_ids (npt.NDArray | None, optional): List of patient IDs to extract. Defaults to all.
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        if not patient_ids:
            patient_ids = self.patient_ids
        f = functools.partial(self._convert_dataset_pt_zip_to_hdf5, zip_path=zip_path, force=force)
        with Pool(processes=num_workers) as pool:
            _ = list(tqdm(pool.imap(f, patient_ids), total=len(patient_ids)))
