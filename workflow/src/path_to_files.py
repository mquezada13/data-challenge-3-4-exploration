from pathlib import Path
# Resolve the workflow directory from this file location
repo_root = Path(__file__).resolve().parents[1]




# Input directories
data_dir = repo_root / "data" / "inputs"

# Orientation data 
orientation_filename= "DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits"
data_ori_path = data_dir / orientation_filename


# Mock dataset: 14 weeks
weekly_files = {
    i: f"dc4_mock_dataset_week_{i}_unbinned_data_filtered_with_SAAcut.fits.gz"
    for i in range(1, 15)
}

weekly_paths = {
    week: data_dir / filename
    for week, filename in weekly_files.items()
}


# Response data
responses = ["Response511.o4.e509_513.s20881894470591.m2555.filtered.nonsparse.binnedimaging.imagingresponse.h5",
"ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5" ]

data_response511_path = data_dir / responses[0]
data_responsecon_path = data_dir / responses[1]

# extended response data 
extended_responses = ["extended_source_response_511_merged.h5", "extended_source_response_continuum_merged.h5"]

data_ext_response511_path = data_dir/extended_responses[0]
data_ext_responsecon_path = data_dir / extended_responses[1]


# Output directory
data_dir_out = repo_root / "data" / "outputs"
plot_dir_out = repo_root / "plot_outputs"

# yaml files

config_dir = repo_root / "config"
config_511_path = config_dir / "dataio_511.yaml"
config_continuum_path = config_dir / "dataio_continuum.yaml"