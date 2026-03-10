#!/bin/python
# type: ignore
import os
import subprocess
from pathlib import Path
import getpass
import argparse

CRED_PERMISSION_LEVEL = 0o600
AIR_API_URL = "https://air.radiology.ucsf.edu/api/"
DEFAULT_PROJECT_ID = "3"
DEFAULT_ANONYMIZATION_PROFILE = "72"

current_dir = Path(__file__).resolve().parent


def get_args():
    """Set up the argument parser and return the parsed arguments."""
    parser = argparse.ArgumentParser(
        description="Command line interface to the UCSF Automated Image Retrieval (AIR) Portal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "accession",
        nargs="?",
        metavar="ACCESSION",
        help=(
            "Accession # to download, or path to csv file with accession #s "
            "in one column."
        ),
    )
    parser.add_argument("-mrn", "--mrn", help="Patient ID to download")
    parser.add_argument(
        "-o", "--output", help="Output path", default=Path.cwd() / "air_download"
    )
    parser.add_argument(
        "-pf",
        "--profile",
        help="Anonymization Profile",
        default=DEFAULT_ANONYMIZATION_PROFILE,
    )
    parser.add_argument(
        "-pj", "--project", help="Project ID", default=DEFAULT_PROJECT_ID
    )
    parser.add_argument(
        "-c",
        "--cred-path",
        help="Login credentials file. If not present, will prompt for AIR_USERNAME and AIR_PASSWORD.",
        default=Path.home() / "air_login.txt",
    )
    parser.add_argument(
        "-lpj",
        "--list-projects",
        action="store_true",
        help="List available project IDs",
    )
    parser.add_argument(
        "-lpf",
        "--list-profiles",
        action="store_true",
        help="List available anonymization profiles",
    )
    parser.add_argument(
        "-xm",
        "--exam_modality_inclusion",
        help=(
            "Comma-separated list of exam modality inclusion patterns (case "
            "insensitive, 'or' logic) for exam . Example: 'MR,CT'"
        ),
        default=None,
    )
    parser.add_argument(
        "-xd",
        "--exam_description_inclusion",
        help=(
            "Comma-separated list of exam description inclusion patterns (case "
            "insensitive, 'or' logic) for exam . Example: 'BRAIN WITH AND WITHOUT CONTRAST'"
        ),
        default=None,
    )
    parser.add_argument(
        "-s",
        "--series_inclusion",
        help=(
            "Comma-separated list of series inclusion patterns (case insensitive, 'or' "
            "logic). Example for T1 type series: 't1,spgr,bravo,mpr'"
        ),
        default=None,
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=None,
        help="Run the container in development mode (i.e., bind the local code) with the path indicated.",
    )
    parser.add_argument(
        "--only-return-accessions",
        action="store_true",
        help="Only return the accessions found for the provided search parameters.",
    )
    return parser.parse_args()


def load_env_variables(file_path: str) -> None:
    """
    Loads environment variables from a .env file into the OS environment.

    Args:
        file_path (str): The path to the .env file.
    """
    env_file = Path(file_path)
    if env_file.is_file():
        with env_file.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Ignore comments
                    key, value = line.split("=", 1)  # Split key and value
                    key, value = key.strip(), value.strip()
                    if key and value:
                        os.environ[key] = value  # Set environment variable


def set_credentials(cred_path=None):
    """Set the AIR_USERNAME and AIR_PASSWORD environment variables."""
    if cred_path is None:
        # Default to checking ~/air_login.txt if no cred_path is provided
        cred_path = Path.home() / "air_login.txt"

    cred_file = Path(cred_path).resolve()

    if cred_file.exists():
        if not cred_file.is_file():
            print(f"'{cred_path}' is not a file.")
            exit(1)

        if oct(cred_file.stat().st_mode)[-3:] != "600":
            print(
                f"Warning: '{cred_path}' does not have read/write-only permissions "
                "for the user (600)."
            )
            try:
                cred_file.chmod(CRED_PERMISSION_LEVEL)
                print(
                    "Permissions changed to read/write-only for the user (600) for "
                    f"'{cred_path}'."
                )
            except Exception as e:
                print(f"Failed to change permissions: {e}")
                exit(1)

        load_env_variables(cred_path)

    else:
        print(f"Credentials file '{cred_path}' not found.")
        username = input("Enter AIR_USERNAME: ")
        password = getpass.getpass("Enter AIR_PASSWORD: ")
        os.environ["AIR_USERNAME"] = username
        os.environ["AIR_PASSWORD"] = password

        # Ask the user if they want to save the credentials
        save_credentials = (
            input(
                "Do you want to save these credentials to a secure file in your "
                "home directory? (important for using this as script) (y/n): "
            )
            .strip()
            .lower()
        )

        if save_credentials == "y":
            try:
                cred_file.touch(CRED_PERMISSION_LEVEL)
                with cred_file.open("w") as f:
                    f.write(f"AIR_USERNAME={username}\n")
                    f.write(f"AIR_PASSWORD={password}\n")

                # Double check the file permissions to 600 (r/w for the user only)
                cred_file.chmod(CRED_PERMISSION_LEVEL)
                print(
                    f"Credentials saved to '{cred_file}' with secure permissions (600)."
                )
            except Exception as e:
                print(f"Warning: Failed to save credentials: {e}")


def get_output_directory(output_path, accession):
    """Determine the output directory based on the provided output path."""
    output_path = Path(output_path)
    if not output_path.is_dir():
        output_path = output_path.parent

    return output_path.resolve()


def create_command(args, output_dir, accession=None):
    command = [
        "apptainer",
        "run",
        "--bind",
        f"{output_dir}:{output_dir}",
    ]
    if args.dev is not None:
        command.extend(
            [
                "--bind",
                f"{args.dev.resolve()}:/app/air-download",
            ]
        )

    command.append(
        f"{current_dir}/air_download2.sif",
    )
    command.append(AIR_API_URL)
    if accession:
        command.append(accession)
    command.extend(
        [
            "-o",
            str(args.output),
            "-pf",
            args.profile,
            "-pj",
            args.project,
        ]
    )

    if args.series_inclusion:
        command.extend(["-s", args.series_inclusion])
    if args.list_projects:
        command.append("-lpj")
    if args.list_profiles:
        command.append("-lpf")
    if args.mrn:
        command.extend(["-mrn", args.mrn])
    if args.exam_modality_inclusion:
        command.extend(["-xm", args.exam_modality_inclusion])
    if args.exam_description_inclusion:
        command.extend(["-xd", args.exam_description_inclusion])
    if args.only_return_accessions:
        command.append("--only-return-accessions")
    # print(" ".join(command))

    return command


def run_container(args):
    """Run the Apptainer container with the provided arguments."""
    if args.accession:
        accession_csv = Path(args.accession)
        if accession_csv.is_file() and accession_csv.exists():
            accession_list = accession_csv.read_text().strip().split("\n")
        else:
            accession_list = [args.accession]

        output_dir = get_output_directory(args.output, args.accession[0])
    else:
        accession_list = []
        output_dir = get_output_directory(args.output, "")

    if accession_list:
        for accession in accession_list:
            command = create_command(args, output_dir, accession)
            print(" ".join(command))
            subprocess.run(command)
    else:
        command = create_command(args, output_dir)
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"\n❌ ERROR running accession {accession}")
            print(result.stderr)
        else:
            if hasattr(args, "accession"):
                print(f"\n✅ SUCCESS for accession {args.accession}")
            else:
                print("\n✅ SUCCESS")


def main():
    args = get_args()
    set_credentials(args.cred_path)
    run_container(args)


if __name__ == "__main__":
    main()
