# NVC_Analysis

This directory contains code to analyze widefield cortical imaging data. Types of analysis include estimation of convolution kernels, regression, and functional connectivity analysis. Published datasets can be found at https://dandiarchive.org/dandiset/001211/draft. The supporting function f_loadNWB.m can be used to load and extract variables from .nwb files.

Used in:
[Rauscher et al. 2024. "Neurovascular Impulse Response Function (IRF) during spontaneous activity differentially reflects intrinsic neuromodulation across cortical regions". bioRXiv.](https://www.biorxiv.org/content/10.1101/2024.09.14.612514v1.full)

Contents:

main.m - main script which analyzes data for main figures

supporting functions - f_2xDeconvolve.m, f_HemCorr.m, f_HemCorrGram.m, f_bpf.m, f_downsample.m, f_estimateIRFalpha.m, f_funConGram.m, f_hemRegress.m, f_smooth2d.m, and f_loadNWB.m.
