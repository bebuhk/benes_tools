#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: error_metrics.py
Author: Bene <bebuhk@ethz.ch>
Created: 2026-08-14
Description: Error metrics for comparing two sets of point (e.g., predicted vs. reference isotherms).
"""

import sys
import numpy as np

# copied from (git: fc-dft-light): s101l /home/bebuhk/master-thesis/fc-dft-light/fcsaft_cdft/plot_isotherms.py 
# USAGE: e.g. mard_mad_mrd_md_calc(pG, lG, lV). (pG isnt needed for error computation but the function still checks its length)
def mard_mad_mrd_md_calc(
    points: np.ndarray,
    experimental_data: np.ndarray,
    model_data: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Calculate the average absolute deviation (AAD) between the experimental data and the model data
    Input:
    points: np.array, experimental_data: np.array, model_data: np.array
    Output:
    MARD%: float (mean absolute relative deviation in percentage: sum(abs(experimental_data - model_data)/experimental_data)/len(points)
    rel_diff: np.array (array of relative differences for each point in %)
    AAD: float (absolute average deviation in percentage: sum(abs(experimental_data - model_data))/len(points)
    abs_diff: np.array (array of absolute differences for each point)
    mard_HR: float (MARD in Henry region (first 5 pressure points, NaN if they don't all exist))
    aad_HR: float (AAD in Henry region (first 5 pressure points, NaN if they don't all exist))
    """
    assert len(experimental_data) == len(
        model_data
    ), "Experimental data and model data should be of the same length"
    assert len(points) == len(
        experimental_data
    ), "Points array should be of the same length as the experimental"
    # sort out NaN values
    assert not np.isnan(
        points
    ).any(), "measurement points should not contain NaN values"
    mard_HR = 0.0
    aad_HR = 0.0
    # get indices of NaN values
    nan_indices = np.argwhere(np.isnan(experimental_data)).flatten()

    # remove NaN values
    experimental_data = np.delete(experimental_data, nan_indices)
    model_data = np.delete(model_data, nan_indices)
    if np.sum(nan_indices) > 0:
        print(
            f"found {len(nan_indices)} NaN values in the experimental data at points: {points[nan_indices]} ({len(nan_indices)}/{len(points)} points)"
        )
    points = np.delete(points, nan_indices)

    nan_indices = np.argwhere(np.isnan(model_data)).flatten()

    # remove NaN values
    experimental_data = np.delete(experimental_data, nan_indices)
    model_data = np.delete(model_data, nan_indices)
    if np.sum(nan_indices) > 0:
        print(
            f"found {len(nan_indices)} NaN values in the model data at points: {points[nan_indices]} ({len(nan_indices)}/{len(points)} points)"
        )
    points = np.delete(points, nan_indices)

    if len(points) == 0:
        print("all points are NaN values. returning NaN values for all metrics")
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # print(f'points after cleaning   : {points}')
    # print(f'cleaned experimental data: {experimental_data}')
    # print(f'cleaned model data: {model_data}')

    diff = (
        model_data - experimental_data
    )  # in bar ## THIS IN NOW data_test - data_ref. (in old code *(-1)...)
    rel_diff_percent = (
        (diff) / abs(experimental_data) * 100
    )  # in % # before 03.12.24: (diff) / experimental_data * 100 # in %
    abs_diff = np.abs(diff)  # in bar (absolute difference)
    mad = np.sum(abs_diff) / len(points)  # mean absolute difference (in bar)
    mard = np.sum(np.abs(rel_diff_percent)) / len(
        points
    )  # ??? but this is mean(ARD) not
    mrd = np.sum((rel_diff_percent)) / len(points)  # mean relative deviation
    md = np.sum((diff)) / len(points)  # mean deviation
    return (
        float(mard),
        float(mad),
        float(mrd),
        float(md),
    )

def main() -> None:
    """Entry point when run as a script."""


if __name__ == "__main__":
    main()