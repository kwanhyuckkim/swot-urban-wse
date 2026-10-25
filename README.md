# swot-urban-wse

Code for my paper on SWOT water surface elevation (WSE) errors in urban rivers:

Kim, K.-H., Gleason, C. J., Andreadis, K. M., & Bennitt, F. B. Geometric Controls on SWOT Water Surface Elevation Measurement in Urban Rivers: Cross-Track Position and Along-Track Flow Angle. IEEE Transactions on Geoscience and Remote Sensing (in review).

I matched SWOT river node WSE with USGS gauges on urban reaches and looked at how the error changes with where the river sits in the swath (near or far from nadir), with the impervious surface and buildings next to it, and with the angle between the river and the satellite track.

## Layout

The code is split by step:

- `pipeline/01_download`: SWORD reach buffers with NLCD land cover, USGS gage heights and their conversion to EGM2008, Hydrocron reach and node data, and the PIXC, PIXCVec and TVP downloads
- `pipeline/02_analysis`: site selection and gauge matchups, the near/far-nadir sampling of impervious surface and building height, the geometry and flow-angle analysis, and an exploratory QRF correction of the WSE error
- `pipeline/03_figures`: notebooks for the figures in the paper, plus plotting cells pulled out of them (`cell_*.py`)

The notebooks have their outputs cleared, and the data and result files are not in the repo.

## Setup

```bash
conda env create -f environment.yml
conda activate swot-urban
```

I ran everything on the Unity cluster at UMass Amherst, so the paths point to our storage there and you will need to change them. The PIXC downloads through earthaccess need a NASA Earthdata login.

## Acknowledgments

This work was supported by the Jet Propulsion Laboratory, California Institute of Technology, under the SWOT Inland Hydrology Calibration and Validation program (Contract No. 1704029) and by NASA under the SWOT and the Global Dynamics of Water Supply, Demand, and Scarcity (Grant No. 80NSSC24K1646). Thanks to the SWOT Science Team for their guidance on data quality and to Merritt Harlan for her help with the SWOT-USGS matching and quality filtering.
