# CRIMAC-reportgeneration-zarr
This repository has been archived and is no longer actively maintained.

The project has moved to a new platform: https://git.imr.no/crimac-wp4-machine-learning/CRIMAC-reportgeneration-zarr

Thank you for your support and contributions! Feel free to check the new repository for the latest updates and discussion.

## Description

This repository provides tools for generating acoustic survey reports by analyzing **acoustic (sv) data**. The main script processes `.zarr` files containing acoustic data, seabed depth information (bottom data), and segmentation model predictions. It applies seabed masking, filters the data based on thresholds, computes aggregated values, and generates reports for further analysis or use in StoX software.

## Repository Structure

```
.
├── LICENSE                 # License file
├── README.md               # Project documentation
├── generate_report.py      # Report generation function
├── script.py               # Main script for report generation
└── requirements.txt        # Required Python packages
```

## Features

- Filters **acoustic (sv) data** based on seabed masking and label thresholds.
- Aggregates filtered **sv data** into `sa` values.
- Generates `.zarr` and `.csv` reports for different segmentation models.

## Input Data

- **sv.zarr**: Acoustic (sv) data from surveys.
- **bottom.zarr**: Seabed depth information for masking.
- **predictions.zarr**: Segmentation model predictions.

### Models and Reports

The reports correspond to predictions from various segmentation models:

| **Report File**  | **Description**                                |
|------------------|-----------------------------------------------|
| `report_1.zarr`  | Report calculated directly from the labels.   |
| `report_2.zarr`  | Report from Brautaset et al. (2020).          |
| `report_3.zarr`  | Report from Ordoñez et al. (2022).            |
| `report_4.zarr`  | Report from Pala et al. (2023).               |

## Requirements

To install the required Python packages, run:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Main Script (`script.py`)

The `script.py` file generates reports for sand eel surveys. It loads **sv data**, **bottom data**, and **segmentation predictions**, applies filtering and masking, and produces reports.

Run the script as follows:

```bash
python script.py
```

### 2. Threshold Definitions

The following thresholds are used for filtering segmentation predictions:

```python
thresholds = {
    "report_1": 1.0,
    "report_2": 0.951172,
    "report_3": 0.900195,
    "report_4": 0.917090,
}
```

### 3. Example Data Structure

The script uses predefined paths for `sv.zarr`, `bottom.zarr`, and predictions. It processes sand eel surveys across multiple years (`cs`) and generates corresponding reports (`pr`).

```python
# Sand eel surveys
cs = ['2005205', '2006207', '2007205', ...]

# Predictions/labels vs reports
pr = [['labels.zarr', 'report_1.zarr'],
      ['predictions_2.zarr', 'report_2.zarr'],
      ['predictions_3.zarr', 'report_3.zarr'],
      ['predictions_4.zarr', 'report_4.zarr']]
```

### 4. Running the Script

For each survey, the script checks for available `sv.zarr`, `bottom.zarr`, and predictions. Missing data is reported, and existing data is processed into reports.

### Output Files

- **`.zarr` reports**: Processed data saved as Zarr format.
- **`.csv` reports**: Tabular reports for StoX software and further analysis.

## Example Workflow

1. **Setup Environment**:
    - Install dependencies using `requirements.txt`.
    - Ensure access to the input data: `sv.zarr`, `bottom.zarr`, and `predictions.zarr`.

2. **Run the Script**:
    ```bash
    python script.py
    ```

3. **Outputs**:
    - Reports are saved as `.zarr` and `.csv` in the specified output directory.

## Notes

- Seabed masking includes a padding of 10 pixels below the seabed.
- Reports are organized by segmentation model and year.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

