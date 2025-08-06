import glob
import os
import argparse
import MDAnalysis as mda
import numpy as np
from pathlib import Path

types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5}

def parse_slice(slice_str: str):
    """Parses a slice string like 'start:stop:step' into a slice object."""
    parts = slice_str.split(':')
    start = None if parts[0] == '' else int(parts[0])
    stop = None if len(parts) < 2 or parts[1] == '' else int(parts[1])
    step = None if len(parts) < 3 or parts[2] == '' else int(parts[2])
    return slice(start, stop, step)

def pair_structures_with_trajectories(structure_files, args_dict):
    """Pairs each structure file with a corresponding trajectory file based on filename stems."""
    input_traj_path = args_dict.get("inputtraj")
    if not input_traj_path:
        return [(str(s), None) for s in structure_files]

    if not os.path.isdir(input_traj_path):
        if len(structure_files) > 1:
            print(f"Warning: A single trajectory file '{input_traj_path.name}' was provided for multiple structures. It will only be paired with the first structure.")
            pairs = [(str(structure_files[0]), str(input_traj_path))]
            pairs.extend([(str(s), None) for s in structure_files[1:]])
            return pairs
        return [(str(structure_files[0]), str(input_traj_path))]

    traj_folder = str(input_traj_path)
    traj_format = args_dict.get("trajformat", "*")
    available_trajs = {
        Path(f).stem: f
        for f in glob.glob(os.path.join(traj_folder, f"**/*.{traj_format}"), recursive=True)
    }

    pairs = []
    for s_file in structure_files:
        s_path = Path(s_file)
        matching_traj = available_trajs.get(s_path.stem)
        if matching_traj:
            print(f"Found matching trajectory for {s_path.name}: {Path(matching_traj).name}")
        pairs.append((str(s_path), matching_traj))
    return pairs

def _process_universe_to_data(universe, args_dict):
    """Helper function to extract coordinate and atom type data from an MDAnalysis Universe."""
    trajslice = args_dict.get("trajslice", None)
    shuffle = args_dict.get("shuffle", True)
    selection = universe.select_atoms(args_dict.get("selection", "all"))
    atom_types = np.array([types[atom.type] for atom in selection.atoms], dtype=np.int8)

    n_frames = len(universe.trajectory)
    frame_indices = np.arange(n_frames)
    if shuffle:
        np.random.shuffle(frame_indices)

    if trajslice is not None:
        frame_indices = frame_indices[parse_slice(trajslice)]

    coords = [selection.positions.copy() for idx in frame_indices if (universe.trajectory[idx], True)]
    return np.array(coords), atom_types

def _save_individual_npz(dataset, input_filename, args_dict, num_input_files):
    """Determines output path based on context and saves a single NPZ dataset."""
    output_arg = args_dict.get("output")
    p_in = Path(input_filename)
    output_path = ""

    if output_arg is None:
        output_path = str(p_in.with_suffix('.npz'))
    else:
        p_out = Path(output_arg)
        if num_input_files > 1:
            output_path = str(p_out / f"{p_in.stem}.npz")
        else:
            output_path = str(p_out.with_suffix('.npz'))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    print(f"Dataset saved: {output_path}")

def run_dataset_processing(args_dict):
    """
    Main function to process structures, and either save them individually
    or join them into a single dataset.
    """
    input_path = args_dict.get("input")
    join_flag = args_dict.get("join")

    # --- Validation ---
    if join_flag:
        if not os.path.isdir(input_path):
            print(f"Error: For the --join operation, the input ('-i') must be a directory.")
            return
        if not args_dict.get("output"):
            print("Error: For the --join operation, an output file path ('-o') is required.")
            return

    # --- File Discovery and Pairing ---
    structure_files = []
    if os.path.isdir(input_path):
        input_format = args_dict.get("inputformat", "*")
        structure_files = list(glob.glob(os.path.join(input_path, f"**/*.{input_format}"), recursive=True))
    elif os.path.exists(input_path):
        structure_files = [str(input_path)]
    
    structure_files = [f for f in structure_files if Path(f).suffix != '.npz']
    
    # --- Main Processing ---
    in_memory_data = []

    if structure_files:
        print("\n--- Processing structure files ---")
        file_pairs = pair_structures_with_trajectories(structure_files, args_dict)
        for structure_file, trajectory_file in file_pairs:
            try:
                print(f"Processing: {Path(structure_file).name}")
                u = mda.Universe(structure_file, trajectory_file) if trajectory_file else mda.Universe(structure_file)
                coords, atom_types = _process_universe_to_data(u, args_dict)
                
                if join_flag:
                    in_memory_data.append((coords, atom_types))
                else:
                    dataset = {"coordinates": coords, "atom_types": atom_types}
                    _save_individual_npz(dataset, structure_file, args_dict, len(file_pairs))
                    
            except Exception as e:
                print(f"Could not process {Path(structure_file).name}. Reason: {e}")
    else:
        print("\n--- No new structure files to process ---")

    # --- Post-Processing (Join Path) ---
    if join_flag:
        print("\n--- Loading existing NPZ datasets ---")
        npz_files = glob.glob(os.path.join(input_path, "**/*.npz"), recursive=True)
        if npz_files:
            for npz_file in npz_files:
                try:
                    print(f"Loading: {npz_file}")
                    with np.load(npz_file) as data:
                        in_memory_data.append((data['coordinates'], data['atom_types']))
                except Exception as e:
                    print(f"Could not load {npz_file}. Reason: {e}")
        else:
            print("No existing .npz files found to load.")

        if not in_memory_data:
            print("\nNo data was built or loaded. Aborting.")
            return

        print("\n--- Merging all datasets ---")
        max_n_atoms = max(d[0].shape[1] for d in in_memory_data)
        total_mols = sum(d[0].shape[0] for d in in_memory_data)
        
        print(f"Creating a merged dataset for {total_mols} total structures with padding up to {max_n_atoms} atoms.")
        merged_coords = np.ma.masked_all((total_mols, max_n_atoms, 3), dtype=np.float32)
        merged_atom_types = np.ma.masked_all((total_mols, max_n_atoms), dtype=np.int8)

        current_mol_idx = 0
        for coords, atom_types in in_memory_data:
            n_frames, n_atoms, _ = coords.shape
            mol_slice = slice(current_mol_idx, current_mol_idx + n_frames)
            atom_slice = slice(0, n_atoms)
            merged_coords[mol_slice, atom_slice, :] = coords
            merged_atom_types[mol_slice, atom_slice] = np.tile(atom_types, (n_frames, 1))
            current_mol_idx += n_frames
            
        dataset = {"coordinates": merged_coords, "atom_types": merged_atom_types}

        # Prepare a new dictionary to hold the arrays and their masks
        save_dict = {}
        for key, masked_array in dataset.items():
            # Add the array itself (np.savez will save its .data attribute)
            save_dict[key] = masked_array 
            # Add the mask as a separate array with the '__mask' suffix
            save_dict[f"{key}__mask__"] = masked_array.mask

        # Get the output path and create the directory if it doesn't exist
        output_path = args_dict.get("output")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the new dictionary containing both data and masks
        np.savez_compressed(output_path, **save_dict)
        print(f"\nJoined dataset saved successfully to: {output_path}")

    print("\nProcessing complete!")

def parse_command_line(args=None):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Build and/or join atomistic datasets.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input path: a single file or a directory.")
    parser.add_argument("-o", "--output", type=Path, help="Output path. Directory for multiple inputs, filename for single input, or required output file for --join.")
    parser.add_argument("--join", action="store_true", help="Builds all structures and joins them with existing .npz files in the input directory.")
    parser.add_argument("-if", "--inputformat", default="*", help="Format of input structure files (e.g., 'pdb').")
    parser.add_argument("-t", "--inputtraj", type=Path, help="Input trajectory file or folder of trajectory files.")
    parser.add_argument("-tf", "--trajformat", default="*", help="Format of trajectory files when -t is a folder (e.g., 'xtc').")
    parser.add_argument("-s", "--selection", default="all", help="MDAnalysis selection string.")
    parser.add_argument("-ts", "--trajslice", type=str, help="Slice of trajectory to process (e.g., '100:2000:10').")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True, help="Shuffle trajectory frames before slicing.")
    return parser.parse_args(args=args)

def main(args=None):
    """Main function to parse arguments and run processing."""
    parsed_args = parse_command_line(args)
    run_dataset_processing(vars(parsed_args))

if __name__ == "__main__":
    main()