# Setting the root path for the imports
from pathlib import Path
import yaml



def write_custom_binning_yaml(
        base_yaml_path,
        output_yaml_path,
        energy_bins,
):

    """
    Create a new binning YAML from an existing configuration.
    parameters
    ----------
    base_yaml_path
        path to the existing YAML configuration
    output_yaml_path
        path where the new YAML will be written
    energy_bins
        Curtom energy-bin edges
    """

    # Normalise the input and output paths
    base_yaml_path = Path(base_yaml_path)
    output_yaml_path = Path(output_yaml_path)

    # Load the existing binning configuration
    with base_yaml_path.open("r") as file:
        config = yaml.safe_load(file)

    # Replace the original energy bins with the custom bin edges
    config["energy_bins"] = [
        float(edge) for edge in energy_bins
    ]

    # Create output directory if does not exist

    output_yaml_path.parent.mkdir(
        parents = True,
        exist_ok = True,
    )

    # write the custom binning configuration
    with output_yaml_path.open("w") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys= False,
        )

    # Return the generated configuration path
    return output_yaml_path