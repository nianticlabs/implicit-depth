import argparse
from pathlib import Path

import yaml


def create_config_and_txt_file(input_dir: Path, save_dir: Path) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)

    # create txt file
    input_dir = input_dir.resolve()
    with open(save_dir / "scans.txt", "w") as f:
        f.write(str(input_dir))

    # create config file
    config = {
        "dataset_path": str(input_dir.parent),
        "tuple_info_file_location": str(save_dir / "tuples"),
        "dataset_scan_split_file": str(save_dir / "scans.txt"),
        "dataset": "vdr",
        "mv_tuple_file_suffix": "_eight_view_deepvmvs_dense.txt",
        "num_images_in_tuple": 8,
        "frame_tuple_type": "dense",
        "split": "test",
    }
    with open(save_dir / "config.yaml", "w") as f:
        f.write("!!python/object:options.Options\n")
        yaml.dump(config, f)

    print(f"Saved config files to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="A small helper script to create a config and txt file for a single sequence for inference"
    )
    parser.add_argument(
        "--input-sequence-dir",
        required=True,
        type=Path,
        help="The path to a VDR sequence folder to process",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        type=Path,
        help="A path to a folder which will be created or overwritten. "
        "A config.yaml and a scans.txt file will be created in this folder.",
    )
    args = parser.parse_args()

    create_config_and_txt_file(input_dir=args.input_sequence_dir, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
