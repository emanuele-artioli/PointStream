"""
A script to run the client-side video reconstruction.
"""
import argparse
from pathlib import Path
from pointstream.client.reconstructor import Reconstructor

def main():
    parser = argparse.ArgumentParser(description="Pointstream Client: Reconstructs video from processed data.")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to the '_final_results.json' file from the server."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the reconstructed scene videos."
    )
    args = parser.parse_args()

    json_path = Path(args.input_json)
    if not json_path.exists():
        print(f"Error: Processed data file not found at {json_path}")
        return

    output_dir = args.output_dir
    if not output_dir:
        input_stem = json_path.stem.replace('_final_results', '')
        output_dir = Path(f"{input_stem}_reconstructed_scenes")
    else:
        output_dir = Path(output_dir)

    reconstructor = Reconstructor(data_path=str(json_path))
    reconstructor.run(output_dir)

if __name__ == "__main__":
    main()