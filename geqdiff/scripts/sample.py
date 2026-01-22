# geqdiff-sample -i /scratch/angiod/qm9/gdb9.sdf --mol_index 0 -d cuda:1 --schedule_type cosine -T 100 --save_trajectory -m /scratch/angiod/GEqDiff/results/foundation/RUN.28.07.25/best_model.pth

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

try:
    import MDAnalysis as mda
    from geqdiff.utils.SDFReader import SDFReader
except ImportError:
    mda = None

from geqdiff.utils.SDFReader import SDFParser
from geqdiff.data import AtomicDataDict
from geqdiff.utils import NoiseScheduler, Sampler, DDPMSampler, DDIMSampler, RectifiedFlowSampler, center_pos
from geqtrain.train.components.checkpointing import CheckpointHandler

# --- Helper functions ---

def load_structure(filepath: str, selection: str = "all", mol_index: int = 0):
    """Loads a structure using MDAnalysis, applies a selection, and extracts data."""
    if mda is None:
        raise ImportError("MDAnalysis is required to read structure files. Please install it: `pip install MDAnalysis`")
    
    try:
        if filepath.lower().endswith(".sdf"):
            print(f"Using SDFReader for {filepath}")
            # Pass the custom class to the 'format' argument.
            # Pass 'mol_index' as a keyword argument to our reader.
            universe = mda.Universe(filepath, format=SDFReader, topology_format=SDFParser, mol_index=mol_index)
        else:
            # Standard MDAnalysis loading for all other formats
            universe = mda.Universe(filepath)
            if len(universe.trajectory) > 1:
                print(f"Warning: Input file {filepath} has multiple frames. Using the frame {mol_index}.")
                universe.trajectory[mol_index]
    except Exception as e:
        raise IOError(f"Could not read structure file {filepath}: {e}")

    atom_group = universe.select_atoms(selection)
    if len(atom_group) == 0:
        raise ValueError(f"The selection '{selection}' resulted in 0 atoms. Please check your selection string.")
    
    print(f"Applied selection '{selection}', using {len(atom_group)} out of {len(universe.atoms)} atoms.")

    positions = torch.tensor(atom_group.positions, dtype=torch.float32)
    try:
        elements = [atom.element for atom in atom_group]
    except:
        elements = [atom.type for atom in atom_group]
    
    element_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8}
    node_types = torch.tensor([element_map.get(el, -1) for el in elements], dtype=torch.long)

    if (node_types == -1).any():
        unknown_elements = set(el for el, nt in zip(elements, node_types) if nt == -1)
        print(f"Warning: Unknown elements found and mapped to -1: {unknown_elements}")

    return center_pos(positions), node_types, elements, atom_group

def build_edge_index(positions: torch.Tensor, cutoff: float):
    """Builds a PyG-style edge_index for a single molecule."""
    dist_matrix = torch.cdist(positions, positions)
    mask = (dist_matrix <= cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index

def build_edge_index_batched(positions: torch.Tensor, cutoff: float):
    """Builds a PyG-style edge_index for a batch of molecules."""
    if positions.dim() != 3:
        raise ValueError("positions must have shape (batch_size, num_atoms, 3).")
    batch_size, num_atoms, _ = positions.shape
    dist_matrix = torch.cdist(positions, positions)
    mask = (dist_matrix <= cutoff) & (dist_matrix > 0)
    edge_idx = mask.nonzero(as_tuple=False)
    if edge_idx.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=positions.device)
    offsets = edge_idx[:, 0] * num_atoms
    row = edge_idx[:, 1] + offsets
    col = edge_idx[:, 2] + offsets
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def save_trajectory_with_mda(output_path: str, atom_group: mda.core.groups.AtomGroup, trajectory_coords: list):
    """Saves a trajectory using MDAnalysis."""
    if mda is None:
        raise ImportError("MDAnalysis is required to save trajectories in XTC/DCD format.")
        
    print(f"Saving trajectory to {output_path}...")
    with mda.Writer(output_path, n_atoms=atom_group.n_atoms) as writer:
        for positions in trajectory_coords:
            atom_group.positions = positions
            writer.write(atom_group)
    print(f"Successfully saved MDAnalysis trajectory to: {output_path}")

# --- Main Sampling Function (Updated) ---

@torch.no_grad()
def sample_from_model(
    model: torch.nn.Module,
    sampler: Sampler,
    device: str,
    r_max: float,
    num_atoms: int,
    node_types: torch.Tensor,
    batch_size: int = 1,
    initial_pos: torch.Tensor = None,
    t_init: int = None,
    steps: int = None,
    ddim_eta: float = 0.0,
    save_trajectory: bool = False,
    field: str = 'noise',
):
    """
    Main function to perform reverse diffusion and generate samples in batch.
    """
    model.to(device)
    model.eval()
    sampler.to(device)

    print(f"Starting sampling on device: {device}")
    print(f"Using sampler: {sampler.__class__.__name__} with r_max = {r_max:.2f}")
    if batch_size > 1:
        print(f"Sampling a batch of {batch_size} molecules.")

    node_types = node_types.to(device)
    batch = torch.arange(batch_size, device=device).repeat_interleave(num_atoms)
    node_types_batch = node_types.repeat(batch_size)

    # 1. Define the starting point x_t
    T_max = sampler.T
    if initial_pos is not None and t_init is not None:
        if t_init >= T_max:
            raise ValueError(f"t_init ({t_init}) must be less than T_max ({T_max})")
        print(f"Starting from provided structure, noising to t={t_init}...")
        
        x_0 = initial_pos.to(device).unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, 3)
        noise = center_pos(torch.randn_like(x_0), batch)
        # Use the sampler's scheduler for the forward process
        alpha_t, sigma_t = sampler.scheduler(torch.tensor([t_init], device=device))
        x_t = center_pos(alpha_t * x_0 + sigma_t * noise, batch)
        start_step = t_init
    else:
        print("Starting from pure random noise...")
        x_t = torch.randn((batch_size * num_atoms, 3), device=device)
        x_t = center_pos(x_t, batch)
        start_step = T_max - 1

    if save_trajectory:
        if batch_size == 1:
            trajectory = [x_t.cpu().numpy()]
        else:
            trajectory = [x_t.reshape(batch_size, num_atoms, 3).cpu().numpy()]
    else:
        trajectory = []

    # 2. Define the timestep sequence for the reverse process
    if isinstance(sampler, (DDIMSampler, RectifiedFlowSampler)):
        if steps is None:
            if isinstance(sampler, DDIMSampler): steps = 50
            elif isinstance(sampler, RectifiedFlowSampler): steps = 20
        time_steps = np.linspace(0, start_step, steps, dtype=int)[::-1].copy()
        print(f"Using {sampler.__class__.__name__} with {len(time_steps)} steps from t={start_step}.")
    else: # DDPMSampler
        time_steps = np.arange(start_step + 1)[::-1].copy()
        print(f"Using DDPM with {len(time_steps)} steps from t={start_step}.")

    # 3. The main reverse diffusion loop

    for i, t in enumerate(tqdm(time_steps, desc="Reverse Diffusion")):
        try:
            # Pre-step check for numerical stability
            if not torch.isfinite(x_t).all():
                print(f"\nWarning: Non-finite coordinates detected at step t={t}. Stopping generation.")
                break

            edge_index = build_edge_index_batched(x_t.reshape(batch_size, num_atoms, 3), r_max)
            data = {
                AtomicDataDict.POSITIONS_KEY: x_t,
                AtomicDataDict.NODE_TYPE_KEY: node_types_batch,
                AtomicDataDict.BATCH_KEY: batch,
                AtomicDataDict.EDGE_INDEX_KEY: edge_index,
                AtomicDataDict.T_SAMPLED_KEY: torch.full((batch_size, 1), t, device=device, dtype=torch.long)
            }

            out_dict = model(data)
            eps_pred = out_dict[field]

            # Call the sampler's step method based on its type
            if isinstance(sampler, DDPMSampler):
                x_t = sampler.step(x_t, t, eps_pred)
            else: # Handles both DDIM and RectifiedFlow
                t_prev = time_steps[i + 1] if i < len(time_steps) - 1 else -1
                if isinstance(sampler, DDIMSampler):
                    x_t = sampler.step(x_t, t, t_prev, eps_pred, eta=ddim_eta)
                else: # RectifiedFlowSampler
                    x_t = sampler.step(x_t, t, t_prev, eps_pred)
                
            x_t = center_pos(x_t, batch)

            if save_trajectory:
                if batch_size == 1:
                    trajectory.append(x_t.cpu().numpy())
                else:
                    trajectory.append(x_t.reshape(batch_size, num_atoms, 3).cpu().numpy())
        except Exception as e:
            print(f"\nAn error occurred during sampling at step t={t}: {e}")
            print("Stopping generation and saving the partial trajectory.")
            break

    print("Sampling complete.")
    return x_t.reshape(batch_size, num_atoms, 3).cpu().numpy(), trajectory

# --- Script Entry Point (Updated) ---

def main(args=None):
    if args is None:
        args = parse_args()

    # --- 1. Load Model and Initial Structure ---
    
    model, config, _ = CheckpointHandler.load_model(args.model, device=args.device)
    initial_pos, node_types, elements, atom_group = load_structure(
        args.input_structure, 
        selection=args.selection, 
        mol_index=args.mol_index
    )
    num_atoms = len(initial_pos)
    
    try:
        r_max = config[AtomicDataDict.R_MAX_KEY]
    except KeyError:
        raise KeyError(f"Could not find '{AtomicDataDict.R_MAX_KEY}' in the model's config.")

    # --- 2. Initialize Scheduler and Sampler ---
    
    scheduler = NoiseScheduler(T=args.Tmax, schedule_type=args.schedule_type)
    
    # Instantiate the chosen sampler
    if args.sampler == "ddpm":
        sampler = DDPMSampler(scheduler)
    elif args.sampler == "ddim":
        sampler = DDIMSampler(scheduler)
    elif args.sampler == "rectified":
        sampler = RectifiedFlowSampler(scheduler)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    # --- 3. Run Sampling ---
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1.")
    batch_size = args.batch_size or args.num_samples
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if batch_size > args.num_samples:
        print(f"Warning: --batch_size ({batch_size}) > --num_samples ({args.num_samples}). Using batch_size = {args.num_samples}.")
        batch_size = args.num_samples

    if args.num_samples > 1:
        print(f"Sampling {args.num_samples} molecules in batches of {batch_size}.")

    final_batches = []
    diffusion_trajectory = None
    for start_idx in range(0, args.num_samples, batch_size):
        current_batch = min(batch_size, args.num_samples - start_idx)
        if args.num_samples > 1:
            print(f"Running batch {start_idx + 1}-{start_idx + current_batch} of {args.num_samples}...")

        final_positions, trajectory = sample_from_model(
            model=model,
            sampler=sampler,
            device=args.device,
            r_max=r_max,
            num_atoms=num_atoms,
            node_types=node_types,
            batch_size=current_batch,
            initial_pos=initial_pos,
            t_init=args.t_init,
            steps=args.steps,
            ddim_eta=args.ddim_eta, # DDIM-specific, ignored by other samplers
            save_trajectory=args.save_trajectory and args.num_samples == 1,
            field=args.field,
        )
        final_batches.append(final_positions)
        if args.save_trajectory and args.num_samples == 1:
            diffusion_trajectory = trajectory

    final_positions = np.concatenate(final_batches, axis=0)

    # --- 4. Save Results ---
    
    if args.output:
        # User provided a path. Use it as the base.
        output_base = Path(args.output)
        # Create parent directory if it doesn't exist
        output_base.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_base.with_suffix('.xyz')
        traj_base_file = output_base
        samples_traj_base = output_base.with_name(output_base.stem + "_samples")
    else:
        # Default behavior: save in sampling_results with auto-generated name
        output_dir = Path("sampling_results")
        output_dir.mkdir(exist_ok=True)
        file_name_stem = f"generated_molecule_{args.sampler}_{args.schedule_type}"
        output_file = output_dir / f"{file_name_stem}.xyz"
        traj_base_file = output_dir / f"trajectory_{args.sampler}_{args.schedule_type}"
        samples_traj_base = output_dir / f"samples_{args.sampler}_{args.schedule_type}"
    
    with open(output_file, "w") as f:
        for sample_idx, sample_positions in enumerate(final_positions):
            f.write(f"{num_atoms}\n")
            if args.num_samples == 1:
                comment = f"Generated by {args.sampler} sampler with {args.schedule_type} schedule. Selection: '{args.selection}'"
            else:
                comment = (f"Sample {sample_idx + 1}/{args.num_samples} generated by {args.sampler} "
                           f"sampler with {args.schedule_type} schedule. Selection: '{args.selection}'")
            f.write(f"{comment}\n")
            for i, pos in enumerate(sample_positions):
                f.write(f"{elements[i]}   {pos[0]:.4f}   {pos[1]:.4f}   {pos[2]:.4f}\n")
            
    print(f"Successfully saved generated coordinates to: {output_file}")

    if args.save_trajectory:
        if args.num_samples == 1:
            traj_file = traj_base_file.with_suffix(f".{args.traj_format}")
            if args.traj_format == "npz":
                np.savez_compressed(traj_file, trajectory=np.array(diffusion_trajectory))
                print(f"Successfully saved trajectory to: {traj_file}")
            else:
                save_trajectory_with_mda(str(traj_file), atom_group, diffusion_trajectory)
        else:
            traj_file = samples_traj_base.with_suffix(f".{args.traj_format}")
            if args.traj_format == "npz":
                np.savez_compressed(traj_file, samples=final_positions)
                print(f"Successfully saved samples trajectory to: {traj_file}")
            else:
                save_trajectory_with_mda(str(traj_file), atom_group, list(final_positions))

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="GeqDiff Sampling Script")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to deployed model or .pth model weights.")
    parser.add_argument("-i", "--input_structure", type=str, required=True, help="Path to an input structure file (PDB, GRO, etc.).")
    parser.add_argument("-o", "--output", type=str, default=None, help="Base path for the output file(s). Suffixes will be added automatically.")
    parser.add_argument("--selection", type=str, default="all", help="MDAnalysis selection string (e.g., 'not name H*').")
    parser.add_argument("--mol_index", type=int, default=0, help="Index of the molecule to use in a multi-molecule file (e.g., SDF). Default is 0.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for sampling. Defaults to --num_samples.")
    parser.add_argument("--t_init", type=int, default=None, help="Timestep to start denoising from. If not set, starts from pure noise.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for sampling.")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim", "rectified"], help="Sampler to use.")
    parser.add_argument("--schedule_type", type=str, default="linear", choices=["linear", "cosine"], help="Noise schedule to use.")
    parser.add_argument("-T", "--Tmax", type=int, default=1000, help="Maximum number of diffusion timesteps.")
    parser.add_argument("-f", "--field", type=str, default="noise", help="Out field where model saves predicted noise.")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps for DDIM or RectifiedFlow sampling.")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="Eta parameter for DDIM sampling (controls stochasticity).")
    parser.add_argument("--save_trajectory", action="store_true", help="Save trajectory. For a single sample, saves the diffusion trajectory. For multiple samples, saves the final samples as a trajectory.")
    parser.add_argument("--traj_format", type=str, default="dcd", choices=["npz", "xtc", "dcd"], help="Format for the saved trajectory. Default: dcd")
    
    if arg_list is not None:
        return parser.parse_args(arg_list)
    else:
        return parser.parse_args()

if __name__ == "__main__":
    main()
