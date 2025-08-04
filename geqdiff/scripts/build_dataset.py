import glob
import os
import argparse
import MDAnalysis as mda
import numpy as np
from pathlib import Path

types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5}

def parse_slice(slice_str: str):
    parts = slice_str.split(':')

    start = None if parts[0] == '' else int(parts[0])
    stop = None if parts[1] == '' else int(parts[1])
    step = None if len(parts) == 2 or parts[2] == '' else int(parts[2])

    return slice(start, stop, step)

def process_structure(input_filename, input_trajname, args_dict):
    extrakwargs = args_dict.get("extrakwargs", {})
    if input_trajname is None:
        input_trajname = []
        u = mda.Universe(input_filename, **extrakwargs)
    else:
        u = mda.Universe(input_filename, input_trajname, **extrakwargs)

    trajslice = args_dict.get("trajslice", None)
    shuffle = args_dict.get("shuffle", True)
    selection = u.select_atoms(args_dict.get("selection", "all"))
    atom_types = np.array([types[atom.type] for atom in selection.atoms], dtype=np.int8)

    # Get all frame indices
    n_frames = len(u.trajectory)
    frame_indices = np.arange(n_frames)
    if shuffle:
        np.random.shuffle(frame_indices)

    # Apply slicing to frame indices
    if trajslice is not None:
        frame_indices = frame_indices[parse_slice(trajslice)]

    coords = []
    for idx in frame_indices:
        u.trajectory[idx]
        coords.append(selection.positions.copy())
    coords = np.array(coords)

    dataset = {
        "coordinates": coords,
        "atom_types": atom_types,
    }

    output_path = args_dict.get("output", None)
    if output_path is None:
        p = Path(input_filename)
        output_path = str(Path(p.parent, p.stem + '.npz'))
    else:
        # If multiple files, append stem to output filename
        if os.path.isdir(output_path):
            p = Path(input_filename)
            output_path = str(Path(output_path).parent / (p.stem + '.npz'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **dataset)
    print(f"Dataset saved: {output_path}")

def build_dataset(args_dict):
    print("Building dataset...")

    input = args_dict.get("input")
    if os.path.isdir(input):
        input_format = args_dict.get("inputformat", "*")
        input_filenames = list(glob.glob(os.path.join(input, f"**/*.{input_format}"), recursive=True))
    else:
        input_filenames = [input]
    
    inputtraj = args_dict.get("inputtraj", None)
    if inputtraj is None:
        input_trajnames = [None for _ in range(len(input_filenames))]
    else:
        input_trajnames = []
        if os.path.isdir(inputtraj):
            traj_format = args_dict.get("trajformat", "*")
            input_stems = [Path(f).stem for f in input_filenames]
            input_trajnames = [
                f if Path(f).stem in input_stems else None
                for f in glob.glob(os.path.join(inputtraj, f"*.{traj_format}"))
            ]
        else:
            assert len(input_filenames) == 1
            input_trajnames = [inputtraj]
    
    for input_filename, input_trajname in zip(input_filenames, input_trajnames):
        process_structure(input_filename, input_trajname, args_dict)
    print("Success!")

def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Build training dataset"
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Either Input folder or Input filename of atomistic structure to map.\n" +
             "Supported file formats are those of MDAnalysis (see https://userguide.mdanalysis.org/stable/formats/index.html)" +
             "If a folder is provided, all files in the folder (optionally filtered, see --filter) with specified extension (see --inputformat) will be used as Input file.",
        type=Path,
        default=None,
        required=True,
    )
    parser.add_argument(
        "-if",
        "--inputformat",
        help="Format of input files to consider, e.g., 'pdb'.\n" +
             "By default takes all formats.",
    )
    parser.add_argument(
        "-t",
        "--inputtraj",
        help="Input trjectory file or folder, which contains all input traj files.",
    )
    parser.add_argument(
        "-tf",
        "--trajformat",
        help="Format of input traj files to consider. E.g. 'trr'. By default takes all formats.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for npz dataset.\n" +
             "If not provided, it will be the folder of the input.\n" +
             "The filename will be the one of the input with the .npz extension.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        help="Selection of atoms to map. Dafaults to 'all'",
        default="all",
    )
    parser.add_argument(
        "-ts",
        "--trajslice",
        help="Specify a slice of the total number of frames.\n" +
             "Only the sliced frames will be backmapped.",
        type=str,
    )
    parser.add_argument(
        "--shuffle",
        help="Shuffle trajectory frames before slicing (default: True).",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser.parse_args(args=args)

def main(args=None):
    args_dict = parse_command_line(args)
    build_dataset(vars(args_dict))

if __name__ == "__main__":
    main()