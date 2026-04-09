# ResultsReader

A Python data processing library made for the cytation 5, but able to process any data in a similar format (See Input File Format). Can be used for autimatically creating time breaks, heat correction of fluorescence values, data normalization, and data visualization (pyplots).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Input File Format](#input-file-format)
- [Quick Start](#quick-start)
- [ResultsReader](#resultsreader-class)
  - [Constructor](#constructor)
  - [Data Access Methods](#data-access-methods)
  - [Data Manipulation Methods](#data-manipulation-methods)
  - [Normalization Methods](#normalization-methods)
  - [Visualization Methods](#visualization-methods)
  - [Utility Methods](#utility-methods)
- [Global Constants](#global-constants)
- [Usage Examples](#usage-examples)
- [Logging](#logging)
---

## Overview

ResultsReader is designed to handle fluorescence data from plate readers, providing:

- Automatic time break detection: Segments data based on gaps in time readings
- Heat correction: Compensates for temperature-dependent fluorescence changes
- Data normalization: Converts raw fluorescence to fractional values using high/low reference points
- Flexible data selection: Query data by time intervals or break indices
- Visualization: Built-in plotting with matplotlib
- Comprehensive logging: Tracks all operations performed on the data

---

## Installation

### Dependencies

    pip install pandas numpy matplotlib jax jaxlib

---

## Input File Format

The input file can be either:
- (Default) A tab delimited table
 - A csv file

In either case, the first column should be "Time" , and subsequent columns should be the well names.

Example:

    Time    A1      A2      B1      B2
    0       1000    1050    980     1020
    10      1010    1060    990     1030
    20      1020    1070    1000    1040
    ...

---

## Quick Start

    from results_reader import ResultsReader

    # Initialize the reader with your data file
    RR = ResultsReader(
        filename="./plate_data.txt",
        outputFolderName="MyExperiment",
        breakSize=120,
        tempIn=37.0,
        tempOut=21.0,
        heatCorrect=True,
        timeUnit="s"
    )

    # View the data structure
    print(RR)

    # Set normalization values
    RR.setLowValuesByTimeInterval(startBound=0, percentile=5)
    RR.setHighValuesByTimeInterval(startBound=1000, percentile=95)

    # Visualize the data
    RR.showDataSeriesByTime()

    # Export processed data
    RR.saveData("processed_data.csv")

---

## ResultsReader Class

### Constructor

#### __init__(filename, outputFolderName, breakSize, tempIn, tempOut, heatCorrect, timeUnit)

Creates a new ResultsReader instance and processes the input data file.

Parameters:

    filename : str (required)
        Path to the tab-delimited plate reader data file
    
    outputFolderName : str (default: "OutputResultsReader")
        Name of the output folder for results and logs
    
    breakSize : int (default: 120)
        Time gap threshold (in time units) to create a new time break
    
    tempIn : float (default: 37.0)
        Temperature inside the plate reader (°C)
    
    tempOut : float (default: 21.0)
        Ambient/room temperature (°C)
    
    heatCorrect : bool (default: True)
        Whether to apply heat correction to fluorescence values
    
    timeUnit : str (default: "s")
        Time unit of the data: "s" (seconds), "m" (minutes), "h" (hours), or "d" (days)

    csv : bool (default: False)
        Whether using a csv or table

Returns:

    ResultsReader instance

Example:

    RR = ResultsReader(
        filename="./experiment_data.txt",
        outputFolderName="Experiment_2024",
        breakSize=120,
        tempIn=37.0,
        tempOut=21.0,
        heatCorrect=True,
        timeUnit="s"
    )

Notes:

    - Creates an output folder containing logs and processed data
    - Automatically removes wells with all-zero data
    - Removes trailing zeros from time series
    - Strips asterisk characters from data

---

### Data Access Methods

#### getWellFrame(well, normedDataBool, fluConcInverse)

Retrieves the complete data frames for a specified well.

Parameters:

    well : str (required)
        Well name (e.g., "A1", "K8")
    
    normedDataBool : bool (default: True)
        If True, returns normalized data (requires high/low values to be set)
    
    fluConcInverse : bool (default: False)
        If True, inverts the normalization (for inverse fluorescence-concentration relationships)

Returns:

    list[pd.DataFrame] - List of DataFrames, one per time break

Example:

    # Get raw data
    raw_data = RR.getWellFrame("K8", normedDataBool=False)

    # Get normalized data
    normalized_data = RR.getWellFrame("K8", normedDataBool=True)

---

#### getWellTimes(well)

Retrieves the time values for a specified well.

Parameters:

    well : str (required)
        Well name

Returns:

    list[pd.Series] - List of time Series, one per time break

Example:

    times = RR.getWellTimes("K8")

---

#### getWellTimesAsJaxList(well)

Retrieves time values as JAX arrays for numerical computation.

Parameters:

    well : str (required)
        Well name

Returns:

    list[jnp.ndarray] - List of JAX arrays containing time values

Example:

    jax_times = RR.getWellTimesAsJaxList("K8")

---

#### getWellData(well, normedData, fluConcInverse)

Retrieves only the fluorescence data values for a specified well.

Parameters:

    well : str (required)
        Well name
    
    normedData : bool (default: True)
        If True, returns normalized data
    
    fluConcInverse : bool (default: False)
        If True, inverts the normalization

Returns:

    list[pd.Series] - List of data Series, one per time break

Example:

    fluorescence = RR.getWellData("K8", normedData=False)

---

#### getWellDataAsJax(well, normedData, fluConcInverse)

Retrieves fluorescence data as a single concatenated JAX array.

Parameters:

    well : str (required)
        Well name
    
    normedData : bool (default: True)
        If True, returns normalized data
    
    fluConcInverse : bool (default: False)
        If True, inverts the normalization

Returns:

    jnp.ndarray - Concatenated JAX array of all fluorescence values

Example:

    jax_data = RR.getWellDataAsJax("K8", normedData=True)

---

### Data Manipulation Methods

#### addTimeBreak(time, columnName)

Manually adds a time break at a specified time point.

Parameters:

    time : int (required)
        Time value at which to create the break
    
    columnName : str (default: "")
        Well name to apply the break to; if empty, applies to all wells

Returns:

    None

Example:

    # Add time break at t=500 for well K8
    RR.addTimeBreak(500, "K8")

    # Add time break at t=1000 for all wells
    RR.addTimeBreak(1000)

---

#### voidTimeSpansByTimeInterval(startBound, endBound, columnName)

Removes data within a specified time range.

Parameters:

    startBound : int | None (default: None)
        Start time of range to void; None = beginning of data
    
    endBound : int | None (default: None)
        End time of range to void; None = end of data
    
    columnName : str | list[str] | None (default: None)
        Well(s) to apply to; None = all wells

Returns:

    None

Example:

    # Remove data between t=360 and t=480 for specific wells
    RR.voidTimeSpansByTimeInterval(startBound=360, endBound=480, columnName=["K8", "K9"])

    # Remove all data before t=240 for well K8
    RR.voidTimeSpansByTimeInterval(endBound=240, columnName="K8")

---

#### voidTimeSpansByIndex(startBound, endBound, columnName)

Removes data within a specified range of time break indices.

Parameters:
    
    startBound : int | None (default: None)
        Start index of breaks to void; None = first break
    
    endBound : int | None (default: None)
        End index of breaks to void; None = last break
    
    columnName : str | list[str] | None (default: None)
        Well(s) to apply to; None = all wells

Returns:

    None

Example:

    # Remove breaks 1 and 2 for specific wells
    RR.voidTimeSpansByIndex(startBound=1, endBound=3, columnName=["K9", "K10"])

---

#### removeWell(columnName)

Completely removes a well from the dataset.

Parameters:

    columnName : str (required)
        Name of the well to remove

Returns:

    None

Example:

    RR.removeWell("K9")

---

### Normalization Methods

#### setHighValuesByTimeInterval(startBound, endBound, columnName, percentile)

Sets the high reference value for normalization based on a percentile within a time range.

Parameters:

    startBound : int | None (default: None)
        Start time of search range; None = beginning
    
    endBound : int | None (default: None)
        End time of search range; None = end
    
    columnName : str | list[str] | None (default: None)
        Well(s) to set; None = all wells
    
    percentile : float (default: 50)
        Percentile value (0-100) to use as high reference

Returns:

    None

Example:

    # Set high value using 95th percentile of data after t=1000
    RR.setHighValuesByTimeInterval(startBound=1000, percentile=95, columnName="K8")

---

#### setLowValuesByTimeInterval(startBound, endBound, columnName, percentile)

Sets the low reference value for normalization based on a percentile within a time range.

Parameters:

    startBound : int | None (default: None)
        Start time of search range; None = beginning
    
    endBound : int | None (default: None)
        End time of search range; None = end
    
    columnName : str | list[str] | None (default: None)
        Well(s) to set; None = all wells
    
    percentile : float (default: 50)
        Percentile value (0-100) to use as low reference

Returns:

    None

Example:

    # Set low value using 5th percentile of first 500 time units
    RR.setLowValuesByTimeInterval(endBound=500, percentile=5)

---

#### setHighValuesByBreakInterval(startBound, endBound, columnName, percentile)

Sets the high reference value based on a percentile within a range of time break indices.

Parameters:

    startBound : int | None (default: None)
        Start break index; None = first break
    
    endBound : int | None (default: None)
        End break index; None = last break
    
    columnName : str | list[str] | None (default: None)
        Well(s) to set; None = all wells
    
    percentile : float (default: 50)
        Percentile value (0-100) to use as high reference

Returns:

    None

Example:

    # Set high value from breaks 1-2 using 100th percentile
    RR.setHighValuesByBreakInterval(startBound=1, endBound=2, percentile=100)

---

#### setLowValuesByBreakInterval(startBound, endBound, columnName, percentile)

Sets the low reference value based on a percentile within a range of time break indices.

Parameters:

    startBound : int | None (default: None)
        Start break index; None = first break
    
    endBound : int | None (default: None)
        End break index; None = last break
    
    columnName : str | list[str] | None (default: None)
        Well(s) to set; None = all wells
    
    percentile : float (default: 50)
        Percentile value (0-100) to use as low reference

Returns:

    None

Example:

    # Set low value from all breaks using 0th percentile (minimum)
    RR.setLowValuesByBreakInterval(percentile=0)

---

#### setHighUsingDifferentWell(settingWell, columnName)

Copies the high reference value from one well to other wells.

Parameters:

    settingWell : str (required)
        Source well with established high value
    
    columnName : str | list[str] | None (default: None)
        Target well(s); None = all wells

Returns:

    None

Example:

    # Use K8's high value for K9 and K10
    RR.setHighUsingDifferentWell("K8", columnName=["K9", "K10"])

---

### Visualization Methods

#### showDataSeriesByTime(startBound, endBound, columnName)

Displays a plot of fluorescence data over a specified time range.

Parameters:

    startBound : int | None (default: None)
        Start time for plot; None = beginning
    
    endBound : int | None (default: None)
        End time for plot; None = end
    
    columnName : str | list[str] | None (default: None)
        Well(s) to plot; None = all wells

Returns:

    None

Notes:

    - Automatically creates two subplots if some wells have high/low values set and others don't
    - Normalized wells appear in the "Normalized Fluorescence" subplot
    - Non-normalized wells appear in the "Fluorescence" subplot

Example:

    # Plot specific wells over a time range
    RR.showDataSeriesByTime(startBound=20000, endBound=40000, columnName=["K8", "K9"])

    # Plot all wells
    RR.showDataSeriesByTime()

---

#### showDataSeriesByIndex(startBound, endBound, columnName)

Displays a plot of fluorescence data for specified time break indices.

Parameters:

    startBound : int | None (default: None)
        Start break index; None = first break
    
    endBound : int | None (default: None)
        End break index; None = last break
    
    columnName : str | list[str] | None (default: None)
        Well(s) to plot; None = all wells

Returns:
    None

Example:

    # Plot first 3 time breaks
    RR.showDataSeriesByIndex(startBound=0, endBound=3)

---

### Utility Methods

#### saveData(filename)

Exports the processed data to a CSV file.

Parameters:

    filename : str | None (default: None)
        Output filename; None defaults to "modifiedData.csv"

Returns:

    None

Example:

    RR.saveData("processed_data.csv")

---

## Global Constants

There are some default global constants defined that are best guesses based on experimental data,
and specific to the materials being used.

kVal = 0.09
    Inverse time constant for temperature equilibration

AmplitudeOvershoot = 0.35
    Fractional fluorescence decrease from heated to room temperature

---

## Usage Examples

### Basic Workflow

    from results_reader import ResultsReader

    # 1. Initialize with your data
    RR = ResultsReader(
        filename="./experiment.txt",
        outputFolderName="Analysis_Output",
        breakSize=120,
        tempIn=37.0,
        tempOut=21.0,
        heatCorrect=True,
        timeUnit="s"
    )

    # 2. Inspect the data structure
    print(RR)

    # 3. Visualize raw data
    RR.showDataSeriesByTime()

    # 4. Remove unwanted data
    RR.voidTimeSpansByTimeInterval(endBound=100)  # Remove first 100 seconds

    # 5. Set normalization references
    RR.setLowValuesByTimeInterval(startBound=100, endBound=500, percentile=5)
    RR.setHighValuesByTimeInterval(startBound=5000, percentile=95)

    # 6. Visualize normalized data
    RR.showDataSeriesByTime()

    # 7. Extract data for further analysis
    normalized_data = RR.getWellDataAsJax("K8", normedData=True)
    time_data = RR.getWellTimesAsJaxList("K8")

    # 8. Save results
    RR.saveData("final_data.csv")

### Working with Multiple Wells

    # Set values for specific wells
    RR.setLowValuesByTimeInterval(
        startBound=0, 
        endBound=500, 
        percentile=5, 
        columnName=["A1", "A2", "A3"]
    )

    # Copy high value from control well to experimental wells
    RR.setHighValuesByTimeInterval(startBound=5000, percentile=95, columnName="Control")
    RR.setHighUsingDifferentWell("Control", columnName=["Sample1", "Sample2"])

    # Plot comparison
    RR.showDataSeriesByTime(columnName=["Control", "Sample1", "Sample2"])

### Handling Time Breaks

    # Add manual time break
    RR.addTimeBreak(1000, "K8")

    # Remove specific time break intervals
    RR.voidTimeSpansByIndex(startBound=1, endBound=2, columnName="K8")

    # View data by break index
    RR.showDataSeriesByIndex(startBound=0, endBound=2)

---

## Logging

ResultsReader maintains detailed logs of all operations:

### Log Structure

    OutputFolder/
    ├── logs/
    │   ├── globalLog.txt      # All operations
    │   ├── wellA1Log.txt      # Well-specific operations
    │   ├── wellA2Log.txt
    │   └── ...
    ├── rawData.txt            # Copy of original input
    └── modifiedData.csv       # Processed output

### Log Format

    1 [1703123456.789] Execution log of MyExperiment.py
    2 [1703123456.790] 2024-01-01 12:00:00
    3 [1703123456.791] Initializing results reader
    ...

Each log entry includes:
- Step number
- Unix timestamp
- Operation description

---
