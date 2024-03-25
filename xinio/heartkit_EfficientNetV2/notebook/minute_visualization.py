# %%
import os
import tensorflow as tf
import plotly.graph_objects as go
import datetime
import random
import numpy as np


from heartkit.tasks import TaskFactory
from typing import Type, TypeVar, List, Dict
from argdantic import ArgField, ArgParser
from pydantic import BaseModel
from heartkit.utils import env_flag, set_random_seed, setup_logger
from plotly.subplots import make_subplots
from tqdm import tqdm
from heartkit.rpc.backends import EvbBackend, PcBackend
from IPython.display import clear_output
from enum import IntEnum


from heartkit.tasks.AFIB_Ident.utils import (
    create_model,
    load_datasets,
    load_test_datasets,
    load_train_datasets,
    prepare,
)


from heartkit.defines import (
    HKDemoParams, HeartBeat, HeartRate, HeartRhythm, HeartSegment
)

from heartkit.tasks.AFIB_Ident.defines import (
    get_class_mapping,
    get_class_names,
    get_class_shape,
    get_classes,
    get_feat_shape,
)

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


def visualize_prediction(fig: go.Figure, ts, x: np.ndarray, start, stop, y_pred: np.ndarray, y_orig: np.ndarray,
                         class_names: List[str], color_dict: Dict[int, str], row_idx: int = 0):
    
    """
    Visualizes a prediction result on a plot figure.

    Args:
        fig (go.Figure): Plotting figure object.
        ts (np.ndarray): Time-stamps of each sample in the input signal.
        x (np.ndarray): Raw ECG signal samples.
        start (int): Index of the first sample in the current time window to visualize. 
        stop (int): Index of the last sample in the current time window to visualize.  
        y_pred (np.ndarray): Predicted labels for each sample within [start..stop].
        y_orig (np.ndarray): Original labels for each sample within [start..stop].
        class_names (List[str]): Class names corresponding to integer label values.
        color_dict (Dict[int,str]): Dictionary mapping integer label values to colors. 
            e.g., {0:'blue', 1:'red', 2:'green'...}
        
    Returns:
         fig : Updated plot figure after adding visualization elements

     Raises:
         ValueError if 'start' or 'stop' indices are out-of-bounds.

    """


    primary_color = "#11acd5"
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
        text=class_names[y_orig[start]],
        showarrow=False,
        row=row_idx,
        col=1,
        font=dict(color=color_dict[y_orig[start]]),
    )

    fig.add_vrect(
        x0=ts[start],
        x1=ts[stop-1] - datetime.timedelta(seconds=0.1),
        # y0=np.max(x)/2,
        y0=0.9,
        # y1=np.max(x)*0.8,  
        y1=1.1,
        fillcolor=color_dict[y_orig[start]],
        opacity=0.25,
        line_width=0,
        row=row_idx,
        col=1,
        secondary_y=False,
    )
    

    if y_pred[start] != y_orig[start]:
        fig.add_vrect(
            x0=ts[start],
            x1=ts[stop-1] - datetime.timedelta(seconds=0.1),
            y0=0.9,
            y1=1.1,
            fillcolor="red",
            opacity=0.25,
            line_width=2,
            line_color="red",
            row=row_idx,
            col=1,
            secondary_y=False,
        )

    # finally add the ECG wave
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
    return fig

def segment_start_end_plot(seg_sig_gen, params, runner, patient_idx, minute_idx):
    """
    Plots a visualization of the predicted and original labels for each 4-second window in a given ECG signal segment.

    Args:
        seg_sig_gen (tuple): Generator object containing input signal data.
        params (ParamsObject): Configuration parameters for prediction.
        runner: Runner object used for model inference and evaluation. 
        patient_idx (int): Index of the target patient in the dataset list.
        minute_idx (int): Index of the target minute to visualize within the selected time segment.

    Returns:
         fig : Updated plot figure after adding visualization elements
         whole_seg_pred: Predicted values for each sample in entire time frame

     Raises:
         ValueError if 'start' or 'stop' indices are out-of-bounds.

    """

    
    bg_color = "rgba(38,42,50,1.0)"
    plotly_template = "plotly_dark"
    color_dict = params.color_dict


    x = seg_sig_gen[0]
    y_sig = seg_sig_gen[1]
    y_pred = np.zeros(x.shape[0], dtype=np.int32)
    segment_id = seg_sig_gen[2]
    class_map = get_class_mapping(params.num_classes)
    class_names = get_class_names(params.num_classes)


    tgt_labels = list(set(class_map.values()))
    tgt_map = {k: class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
    y_sig = y_sig[np.where(~np.isin(y_sig[:, 1], [IcentiaRhythm.noise.value, IcentiaRhythm.end.value]))] # filter the noise and end
    y_orig = np.vectorize(tgt_map.get, otypes=[int])(y_sig[:, 1]) # from 0-4 to 0-3

    if len(y_orig) == 0:
        print("Unidentified label")
        y_orig = np.full(x.shape[0], -1)
    elif len(y_orig) == 1:
        y_orig = np.full(x.shape[0], y_orig[0])
    else: # a more complicated cases where you have AFib mixed with AFlut
        # let's do majority voting here
        print("Majority voting for multi-rlabel case")
        y_orig = np.full(x.shape[0], np.argmax(np.bincount(y_orig)))
        
    n_sample = x.shape[0] / params.frame_size
    nrow = int(n_sample/5)
    # print(f"Total number of 4 seconds: {n_sample} and rows of plots {nrow}")
    runner.open()
    tod = datetime.datetime(2024, 5, 24, random.randint(12, 23), 00)
    ts = np.array([tod + datetime.timedelta(seconds=i / params.sampling_rate) for i in range(x.shape[0])])
    whole_seg_pred = []
    row_idx = 0
    
    fig = make_subplots(
            rows=nrow,
            cols=1,
            specs=[[{"colspan": 1, "type": "xy", "secondary_y": True}]] * nrow,
            subplot_titles=(None, None),
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
    )

    for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
        # ratios = []
        if i % (5*params.frame_size) == 0:
            row_idx += 1
        if i + params.frame_size > x.shape[0]:
            start, stop = x.shape[0] - params.frame_size, x.shape[0]
        else:
            start, stop = i, i + params.frame_size
        xx = prepare(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        # y_orig[start:stop] = 
        # this is the predicted label for current frame
        y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
        if y_orig[start] == -1:
            print(f"We should skip this sample")
            whole_seg_pred.append(-1)
        # Assuming y_pred and y_orig are numpy arrays
        elif y_pred[start] == y_orig[start]:
            whole_seg_pred.append(1)
        else:
            whole_seg_pred.append(0)
        # whole_seg_pred.append(ratios)
    
        fig = visualize_prediction(
            fig,
            ts,
            x,
            start,
            stop,
            y_pred, 
            y_orig,
            class_names,
            color_dict,
            row_idx,
        )
    fig.update_layout(
        template=plotly_template,
        height=400*nrow,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        legend=dict(groupclick="toggleitem"),
        title=f"Patient ID: {patient_idx}, Segment ID: {segment_id}, Minute: {minute_idx}",
        title_x=0.5,
    )
    # fig.write_html(params.job_dir / "longer_demo.html", include_plotlyjs="cdn", full_html=True)
    # fig.show()

    return fig, whole_seg_pred


def whole_sample_predict(seg_sig_gen, params, runner):
    """
    This function performs prediction on the whole sample of time series data.

    Parameters:
    seg_sig_gen (tuple): A tuple containing the time series data, the original labels, and other information.
    params (object): An object that contains various parameters including frame size, color dictionary, etc.
    runner (object): An object that is used to perform inference.

    Returns:
    x (numpy array): The time series data.
    y_pred (numpy array): The predicted labels for the time series data.
    y_orig (numpy array): The original labels for the time series data.

    The function first prepares the original labels and initializes the predicted labels. Then, it iterates over the time series data in chunks of size equal to the frame size. For each chunk, it prepares the data, performs inference, and updates the predicted labels. It also updates the `whole_seg_pred` list based on the comparison between the predicted and original labels. Finally, it returns the time series data, the predicted labels, and the original labels.
    """
        
    x = seg_sig_gen[0]
    y_sig = seg_sig_gen[1]
    y_pred = np.zeros(x.shape[0], dtype=np.int32)

    class_map = get_class_mapping(params.num_classes)

    tgt_map = {k: class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
    y_sig = y_sig[np.where(~np.isin(y_sig[:, 1], [IcentiaRhythm.noise.value, IcentiaRhythm.end.value]))] # filter the noise and end
    y_orig = np.vectorize(tgt_map.get, otypes=[int])(y_sig[:, 1]) # from 0-4 to 0-3

    if len(y_orig) == 0:
        print("Unidentified label")
        y_orig = np.full(x.shape[0], -1)
    elif len(y_orig) == 1:
        y_orig = np.full(x.shape[0], y_orig[0])
    else: # a more complicated cases where you have AFib mixed with AFlut
        # let's do majority voting here
        print("Majority voting for multi-rlabel case")
        y_orig = np.full(x.shape[0], np.argmax(np.bincount(y_orig)))
        

    runner.open()

    whole_seg_pred = []

    
    for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
        if i + params.frame_size > x.shape[0]:
            start, stop = x.shape[0] - params.frame_size, x.shape[0]
        else:
            start, stop = i, i + params.frame_size
        xx = prepare(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        # this is the predicted label for current frame
        y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
        if y_orig[start] == -1:
            print(f"We should skip this sample")
            whole_seg_pred.append(-1)
        # Assuming y_pred and y_orig are numpy arrays
        elif y_pred[start] == y_orig[start]:
            whole_seg_pred.append(1)
        else:
            whole_seg_pred.append(0)
            
    return x, y_pred, y_orig, whole_seg_pred


def plot_ts(x, params, y_pred, y_orig, sample_meta):
    """
    This function plots the time series data with the original and predicted labels by the requested patient_id, segment_id, minute_id.

    Parameters:
    x (numpy array): The time series data to be plotted.
    params (object): An object that contains various parameters including frame size, color dictionary, etc.
    y_pred (numpy array): The predicted labels for the time series data.
    y_orig (numpy array): The original labels for the time series data.
    sample_meta (list): A list containing metadata about the sample. It includes patient ID, segment ID, and minute.

    Returns:
    fig (plotly.graph_objs._figure.Figure): A Plotly figure object that contains the plot of the time series data.

    The function first prepares the layout for the plot. Then, it iterates over the time series data in chunks of size equal to the frame size. For each chunk, it visualizes the prediction and updates the figure. Finally, it updates the layout of the figure and returns it.
    """
        
    bg_color = "rgba(38,42,50,1.0)"
    plotly_template = "plotly_dark"
    color_dict = params.color_dict

    class_names = get_class_names(params.num_classes)

    n_sample = x.shape[0] / params.frame_size
    nrow = int(n_sample/5)

    tod = datetime.datetime(2024, 5, 24, random.randint(12, 23), 00)
    ts = np.array([tod + datetime.timedelta(seconds=i / params.sampling_rate) for i in range(x.shape[0])])
    row_idx = 0

    # prepare the plot template
    fig = make_subplots(
            rows=nrow,
            cols=1,
            specs=[[{"colspan": 1, "type": "xy", "secondary_y": True}]] * nrow,
            subplot_titles=(None, None),
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
    )

    # traverse through every frame
    for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
        if i % (5*params.frame_size) == 0:
            row_idx += 1
        if i + params.frame_size > x.shape[0]:
            start, stop = x.shape[0] - params.frame_size, x.shape[0]
        else:
            start, stop = i, i + params.frame_size
        # key visualization helper function
        fig = visualize_prediction(
            fig,
            ts,
            x,
            start,
            stop,
            y_pred, 
            y_orig,
            class_names,
            color_dict,
            row_idx,
        )
    
    fig.update_layout(
        template=plotly_template,
        height=400*nrow,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        legend=dict(groupclick="toggleitem"),
        title=f"Patient ID: {sample_meta[0]}, Segment ID: {sample_meta[1]}, Minute: {sample_meta[2]}",
        title_x=0.5,
    )

    return fig

def long_sample_minutes(datasets, patient_idx, segment_idx, minute_idx, params, runner):
    """
    Make a long-term prediction for a specific patient's ECG signal at a given minute index with plot

    Args:
        datasets (list): List of SignalMetaGenerator objects containing ECG data.
        patient_idx (int): Index of the target patient in the dataset list. [0:10999]
        segment_idx (int): Index of the target time segment in the selected patient's data. [0:49]
        minute_idx (int): Index of the target minute to predict within the selected time segment. [0:59]
        params (ParamsObject): Configuration parameters for prediction. [config files]
        runner: Runner object used for model inference and evaluation. 


    Returns:
        whole_seg_pred: Predicted values for each sample in the entire time frame. 

    Raises:
        ValueError: If 'minute_idx' is outside valid range [0-59].
    """
    if minute_idx not in range(60):
        raise ValueError("minute_idx must be between 0 and 59")
    # input_size = int(np.round((params.sampling_rate / params.target_rate) * params.frame_size))
    x_start = 60 * params.target_rate * minute_idx
    x_end = 60 * params.target_rate * (minute_idx+1)


    # input should be a SignalMetaGenerator
    single_pat_gen = datasets[0].uniform_patient_generator(patient_ids=[patient_idx], repeat=False, shuffle=False)
    seg_sig_gen = datasets[0].signal_label_TimeFrame_generator(single_pat_gen, segment_idx=segment_idx, frame_start=int(x_start), frame_end=int(x_end))
    seg_sig_gen = next(seg_sig_gen)
    # print(seg_sig_gen[0].shape[0])
    whole_seg_pred = segment_start_end_plot(seg_sig_gen, params, runner, patient_idx, minute_idx)
    whole_seg_pred = np.array(whole_seg_pred)
    non_neg = whole_seg_pred[whole_seg_pred >= 0]
    print(f"Current time prediction accuracy: {np.sum(non_neg) / len(non_neg)}")
    return whole_seg_pred


def long_prediction(datasets, patient_idx, segment_idx, minute_idx, params, runner, plotting=False):
    """
    Make a long-term prediction for a specific patient's ECG signal at a given minute index.

    Args:
        datasets (list): List of SignalMetaGenerator objects containing ECG data.
        patient_idx (int): Index of the target patient in the dataset list. [0:10999]
        segment_idx (int): Index of the target time segment in the selected patient's data. [0:49]
        minute_idx (int): Index of the target minute to predict within the selected time segment. [0:59]
        params (ParamsObject): Configuration parameters for prediction. [config files]
        runner: Runner object used for model inference and evaluation. 
        plotting (bool, optional): Flag indicating whether to plot results. Defaults to False.

    Returns:
        whole_seg_pred: Predicted values for each sample in the entire time frame. 
        fig: Plot figure if 'plotting' flag is set to True

    Raises:
        ValueError: If 'minute_idx' is outside valid range [0-59].
    """

    if minute_idx not in range(60):
        raise ValueError("minute_idx must be between 0 and 59")
    # input_size = int(np.round((params.sampling_rate / params.target_rate) * params.frame_size))
    x_start = 60 * params.target_rate * minute_idx
    x_end = 60 * params.target_rate * (minute_idx+1)

    # Create generator from input dataset
    single_pat_gen = datasets[0].uniform_patient_generator(patient_ids=[patient_idx], repeat=False, shuffle=False)
    # Generate signals and labels for current time frame
    seg_sig_gen = datasets[0].signal_label_TimeFrame_generator(single_pat_gen,
                                                                segment_idx=segment_idx,
                                                                frame_start=int(x_start),
                                                                frame_end=int(x_end))
    seg_sig_gen = next(seg_sig_gen)
    sample_meta = [patient_idx, segment_idx, minute_idx]

    # Perform whole-sample prediction on generated signals/labels
    x, y_pred, y_orig, whole_seg_pred = whole_sample_predict(seg_sig_gen, params, runner)
    # We should also get the tru AFIB/AFLUT patients from the y_orig
    # y_orig == 1
    whole_seg_pred = np.array(whole_seg_pred)
    non_neg = whole_seg_pred[whole_seg_pred >= 0]
    print(f"Current time prediction accuracy: {np.sum(non_neg) / len(non_neg)}")
    # return the prediction matrix with fig
    if plotting == True:
        fig = plot_ts(x, params, y_pred, y_orig, sample_meta)
        return whole_seg_pred, fig

    return whole_seg_pred

