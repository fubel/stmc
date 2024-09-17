import configparser
import os
import pathlib
from typing import Dict, List, Optional, Union

import motmetrics as mm
import numpy as np
import pandas as pd
import torch
from sklearn import metrics


GT_COLUMNS = [
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
]


def get_hota_setup():
    metrics = ["deta_alpha", "assa_alpha", "hota_alpha"]
    namemap = mm.io.motchallenge_metric_names
    namemap.update({"hota_alpha": "HOTA", "assa_alpha": "ASSA", "deta_alpha": "DETA"})
    return metrics, namemap


def evaluate_tracker(tracker_results, dataloader, hota_mode=False, bev_mode=False):
    gt_dfs = [pd.DataFrame(gt, columns=GT_COLUMNS) for gt in dataloader.dataset._ground_truths]
    ht_dfs = results_to_dfs(tracker_results)

    n_frames = [int(df["frame"].max()) for df in gt_dfs]

    gt_dfs = [mot_to_mm(df) for df in gt_dfs]
    ht_dfs = [mot_to_mm(df) for df in ht_dfs]

    gt_df = combine_dataframes(gt_dfs, n_frames)
    ht_df = combine_dataframes(ht_dfs, n_frames)

    # put column "x" to "X"
    if bev_mode:
        ht_df["X"] = ht_df["x"]
        ht_df["Y"] = ht_df["y"]
        gt_df["X"] = gt_df["x"]
        gt_df["Y"] = gt_df["y"]

    return evaluate_single_scene(ht_df, gt_df, hota_mode=hota_mode, bev_mode=bev_mode)


def results_to_dfs(tracker_results: torch.Tensor) -> List[pd.DataFrame]:
    """Converts a tensor of results to a list of dataframes. Input tensor has format

        CAM_ID, OBJ_ID, FRAME_ID, X, Y, W, H, X_WORLD, Y_WORLD

    and resulting (n_cams) dataframes have columns

        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

    Args:
        tracker_results (torch.Tensor): Results tensor.
    Returns:
        List[pd.DataFrame]: List of dataframes.
    """
    results = tracker_results.clone()
    results[:, [1, 2]] = results[:, [2, 1]]
    results = torch.cat((results[:, :7], torch.ones(results.shape[0], 1), results[:, 7:]), dim=1)
    results = torch.cat((results, -torch.ones(results.shape[0], 1)), dim=1)
    cam_res = [results[results[:, 0] == c][:, 1:] for c in torch.unique(results[:, 0]).cpu().numpy()]
    return [pd.DataFrame(res, columns=GT_COLUMNS) for res in cam_res]


def evaluate_multi_scene(prediction_dfs, ground_truth_dfs, names=None, hota_mode=False, bev_mode=False):
    """Takes prediction and ground truth dataframes and runs motmetrics evaluation
    on a multiple scenes. For evaluation of multi-camera scenes, first combine a
    list of single-camera predictions and ground truths using `combine_dataframes`
    Args:
        prediction_dfs (_type_): _description_
        ground_truth_dfs (_type_): _description_
        names (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: _description_
    """
    if names is None:
        names = ["Untitled %s" % (i + 1) for i in range(len(prediction_dfs))]
    ground_truths = dict(zip(names, ground_truth_dfs))
    predictions = dict(zip(names, prediction_dfs))
    accs = []
    names = []

    if bev_mode:
        distfields = ["X", "Y"]
        dist = "seuc"
        distth = 1.0
    else:
        distfields = ["X", "Y", "Width", "Height"]
        dist = "iou"
        distth = 0.5

    for name, prediction in predictions.items():
        if hota_mode:
            raise NotImplementedError
        else:
            accs.append(
                mm.utils.compare_to_groundtruth(
                    ground_truths[name], prediction, dist=dist, distfields=distfields, distth=distth
                )
            )
            metrics = mm.metrics.motchallenge_metrics
            namemap = mm.io.motchallenge_metric_names
        names.append(name)

    mh = mm.metrics.create()

    summary = mh.compute_many(
        accs,
        names=names,
        metrics=metrics,
        generate_overall=True,
    )
    namemap.update({"hota_alpha": "HOTA", "assa_alpha": "ASSA", "deta_alpha": "DETA"})
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=namemap))
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=namemap)
    return summary, strsummary


def evaluate_single_scene(prediction_df, ground_truth_df, hota_mode=False, bev_mode=False, name=None) -> pd.DataFrame:
    """Takes a prediction and ground truth dataframe and runs motmetrics evaluation
    on a single scene. For evaluation of multi-camera scenes, first combine a list
    of single-camera predictions and ground truths using `combine_dataframes`.
    Args:
        prediction_df (_type_): Multi-camera predictions.
        ground_truth_df (_type_): Multi-camera ground truth.
        name (str): Scene name. Defaults to None.
    """
    return evaluate_multi_scene([prediction_df], [ground_truth_df], [name], hota_mode, bev_mode)


def mot_to_mm(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a MOT-style dataframe (with named columns [frame, id, ...])
    and converts it to a dataframe with column names required by motmetrics.
    Args:
        df (pd.DataFrame): Input MOT-style dataframe.
    Returns:
        pd.DataFrame: Output dataframe ready to use in motmetrics evaluation.
    """
    _df = df.rename(
        columns={
            "frame": "FrameId",
            "id": "Id",
            "bb_left": "X",
            "bb_top": "Y",
            "bb_width": "Width",
            "bb_height": "Height",
            "conf": "Confidence",
        }
    )
    columns_to_int = ["FrameId", "Id", "X", "Y", "Width", "Height"]
    columns_to_float = ["Confidence"]
    _df[columns_to_int] = _df[columns_to_int].astype(int)
    _df[columns_to_float] = _df[columns_to_float].astype(float)
    return _df


def read_txt(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    _df = pd.read_csv(path, names=GT_COLUMNS)
    _df = _df.rename(
        columns={
            "frame": "FrameId",
            "id": "Id",
            "bb_left": "X",
            "bb_top": "Y",
            "bb_width": "Width",
            "bb_height": "Height",
            "conf": "Confidence",
        }
    )
    columns_to_int = ["FrameId", "Id", "X", "Y", "Width", "Height"]
    columns_to_float = ["Confidence"]
    _df[columns_to_int] = _df[columns_to_int].astype(int)
    _df[columns_to_float] = _df[columns_to_float].astype(float)
    return _df


def read_seqinfo(path: Union[str, pathlib.Path]) -> Dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    return dict(parser["Sequence"])


def combine_dataframes(dataframes: List[pd.DataFrame], n_frames: Optional[List[int]] = None) -> pd.DataFrame:
    """Takes a list of single-camera dataframes and combines them for
    multi-camera evaluation.
    Args:
        dataframes (List[pd.DataFrame]): List of single-camera dataframes.
        n_frames (Optional[List[int]], optional): Defaults to None.
    Returns:
        pd.DataFrame: Multi-camera dataframe.
    """
    if n_frames is None:
        n_frames = [int(df["FrameId"].max()) for df in dataframes]
    count_frames = 0
    dfs = []
    for j, df in enumerate(dataframes):
        df["FrameId"] += count_frames
        count_frames += int(n_frames[j])
        dfs.append(df)
    return pd.concat(dfs).set_index(["FrameId", "Id"])


def evaluate_mtmc(
    data_paths: List[Union[str, pathlib.Path]],
    prediction_path: Union[str, pathlib.Path],
    scene_name: str,
    hota_mode=False,
    bev_mode=False,
):
    seqinfos = [read_seqinfo(os.path.join(path, "seqinfo.ini")) for path in data_paths]
    ground_truths = [read_txt(os.path.join(path, "gt", "gt.txt")) for path in data_paths]
    prediction_paths = [os.path.join(prediction_path, seqinfo["name"] + ".txt") for seqinfo in seqinfos]
    predictions = [read_txt(path) for path in prediction_paths]
    ground_truth_df = combine_dataframes(ground_truths, [seqinfo["seqlength"] for seqinfo in seqinfos])
    prediction_df = combine_dataframes(predictions, [seqinfo["seqlength"] for seqinfo in seqinfos])

    ground_truths = {scene_name: ground_truth_df}
    predictions = {scene_name: prediction_df}


def evaluate_synthehicle_json(prediction, ground_truth):
    preds_to_eval = []
    truths_to_eval = []
    names = []
    for scene in ground_truth.keys():
        if scene in prediction.keys():
            gcams = ground_truth[scene]
            pcams = prediction[scene]
            preds_to_combine = []
            truths_to_combine = []
            for cam in gcams.keys():
                if cam not in pcams.keys():
                    prediction[scene][cam] = [[1, 1, 0, 0, 0, 0, 1, -1, -1, -1]]
                preds_to_combine.append(mot_to_mm(pd.DataFrame(prediction[scene][cam], columns=GT_COLUMNS)))
                truths_to_combine.append(mot_to_mm(pd.DataFrame(ground_truth[scene][cam], columns=GT_COLUMNS)))
            names.append(scene)
            preds_to_eval.append(combine_dataframes(preds_to_combine, n_frames=[1800] * len(preds_to_combine)))
            truths_to_eval.append(combine_dataframes(truths_to_combine, n_frames=[1800] * len(truths_to_combine)))
    return evaluate_multi_scene(preds_to_eval, truths_to_eval, names)


def clustering_performance(y_true, y_pred):
    y_t, y_p = y_true.cpu().numpy(), y_pred.cpu().numpy()
    return {
        "ARI": metrics.adjusted_rand_score(y_t, y_p),
        "AMI": metrics.adjusted_mutual_info_score(y_t, y_p),
    }
