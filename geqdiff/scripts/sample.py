# geqdiff-sample -i /scratch/angiod/qm9/gdb9.sdf --mol_index 0 -d cuda:1 --schedule_type cosine -T 100 --save_trajectory -m /scratch/angiod/GEqDiff/results/foundation/RUN.28.07.25/best_model.pth

from typing import Optional
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
from geqdiff.utils import (
    DDPMSampler,
    DDIMSampler,
    FlowMatchingSampler,
    FlowMatchingScheduler,
    NoiseScheduler,
    RectifiedFlowSampler,
    Sampler,
    center_pos,
)
from geqtrain.train.components.checkpointing import CheckpointHandler

# --- Helper functions ---

def load_structure(
    filepath: str,
    selection: str = "all",
    mol_index: int = 0,
    noh: bool = False,
    node_types_map: Optional[list] = None,
):
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

    if noh:
        selection = f"({selection}) and not name H*"
    atom_group = universe.select_atoms(selection)
    if len(atom_group) == 0:
        raise ValueError(f"The selection '{selection}' resulted in 0 atoms. Please check your selection string.")
    
    print(f"Applied selection '{selection}', using {len(atom_group)} out of {len(universe.atoms)} atoms.")

    positions = torch.tensor(atom_group.positions, dtype=torch.float32)
    try:
        elements = [atom.element for atom in atom_group]
    except Exception:
        elements = [atom.type for atom in atom_group]

    if node_types_map:
        element_map = {name: idx for idx, name in enumerate(node_types_map)}
        node_types_idx = []
        for el in elements:
            idx = element_map.get(el)
            if idx is None:
                idx = element_map.get(el.capitalize())
            if idx is None:
                idx = element_map.get(el.upper())
            node_types_idx.append(-1 if idx is None else idx)
    else:
        if noh:
            element_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7}
        else:
            element_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8}
        node_types_idx = [element_map.get(el, -1) for el in elements]

    node_types = torch.tensor(node_types_idx, dtype=torch.long)

    if (node_types == -1).any():
        unknown_elements = set(el for el, nt in zip(elements, node_types.tolist()) if nt == -1)
        raise ValueError(f"Unknown elements found in structure: {sorted(unknown_elements)}")

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


def make_batch_from_atom_counts(atom_counts: torch.Tensor, device: str):
    atom_counts = atom_counts.to(device=device, dtype=torch.long)
    batch_size = int(atom_counts.shape[0])
    return torch.repeat_interleave(torch.arange(batch_size, device=device), atom_counts)


def split_by_atom_counts(values, atom_counts: torch.Tensor):
    parts = []
    offset = 0
    for count in atom_counts.tolist():
        next_offset = offset + int(count)
        parts.append(values[offset:next_offset])
        offset = next_offset
    return parts


def pad_sample_arrays(sample_arrays, atom_counts: torch.Tensor, pad_value):
    if len(sample_arrays) == 0:
        return np.empty((0, 0, 0), dtype=np.float32)

    max_atoms = int(atom_counts.max())
    example = sample_arrays[0]
    out_shape = (len(sample_arrays), max_atoms) + tuple(example.shape[1:])
    if isinstance(pad_value, float) and np.isnan(pad_value):
        dtype = np.result_type(example.dtype, np.float32)
    else:
        dtype = example.dtype
    out = np.full(out_shape, pad_value, dtype=dtype)
    for idx, (arr, count) in enumerate(zip(sample_arrays, atom_counts.tolist())):
        out[idx, :int(count)] = arr
    return out


def pad_batched_atom_axis(array: np.ndarray, target_atoms: int, pad_value):
    if array.shape[2] >= target_atoms:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[2] = (0, target_atoms - array.shape[2])
    return np.pad(array, pad_width=pad_width, mode="constant", constant_values=pad_value)


def build_edge_index_from_atom_counts(positions: torch.Tensor, atom_counts: torch.Tensor, cutoff: float):
    """Builds an edge_index for a flat tensor of nodes grouped by atom_counts."""
    if positions.dim() != 2:
        raise ValueError("positions must have shape (num_nodes, 3).")
    edge_index_parts = []
    offset = 0
    for count in atom_counts.tolist():
        count = int(count)
        pos_i = positions[offset:offset + count]
        edge_i = build_edge_index(pos_i, cutoff)
        if edge_i.numel() > 0:
            edge_index_parts.append(edge_i + offset)
        offset += count
    if len(edge_index_parts) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=positions.device)
    return torch.cat(edge_index_parts, dim=1)


def infer_node_type_metadata(config, node_types_map_arg: Optional[str]):
    if node_types_map_arg:
        node_types_map = [name.strip() for name in node_types_map_arg.split(",")]
    else:
        cfg_type_names = config.get("type_names")
        node_types_map = list(cfg_type_names) if cfg_type_names is not None else None

    cfg_num_types = config.get("num_types")
    num_types = int(cfg_num_types) if cfg_num_types is not None else None
    if num_types is None and node_types_map is not None:
        num_types = len(node_types_map)
    return node_types_map, num_types


def infer_model_Tmax(config):
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    if isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict):
        tmax = stack[0].get("Tmax")
        if tmax is not None:
            return int(tmax)
    return None

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
    atom_counts: torch.Tensor,
    node_types: torch.Tensor,
    initial_pos: torch.Tensor = None,
    t_init: int = None,
    steps: int = None,
    ddim_eta: float = 0.0,
    save_trajectory: bool = False,
    save_npz: bool = False,
    field: Optional[str] = None,
    predict_node_types_field: Optional[str] = None,
):
    """
    Main function to perform reverse diffusion and generate samples in batch.
    """
    model.to(device)
    model.eval()
    sampler.to(device)

    if field is None:
        field = "velocity" if isinstance(sampler, FlowMatchingSampler) else "noise"

    print(f"Starting sampling on device: {device}")
    print(f"Using sampler: {sampler.__class__.__name__} with r_max = {r_max:.2f}")
    atom_counts = atom_counts.to(dtype=torch.long)
    batch_size = int(atom_counts.shape[0])
    total_nodes = int(atom_counts.sum().item())
    atom_counts_cpu = atom_counts.cpu()
    if batch_size > 1:
        print(f"Sampling a batch of {batch_size} molecules.")

    node_types = node_types.to(device)
    batch = make_batch_from_atom_counts(atom_counts, device=device)
    node_types_batch = node_types.to(device)
    if int(node_types_batch.shape[0]) != total_nodes:
        raise ValueError(
            f"node_types has {int(node_types_batch.shape[0])} entries but atom_counts requires {total_nodes} nodes."
        )

    # 1. Define the starting point x_t
    T_max = sampler.T
    if initial_pos is not None and t_init is not None:
        if t_init >= T_max:
            raise ValueError(f"t_init ({t_init}) must be less than T_max ({T_max})")
        if atom_counts.unique().numel() != 1:
            raise ValueError("A provided input structure requires all samples in the batch to have the same atom count.")
        if isinstance(sampler, FlowMatchingSampler):
            print(f"Starting from provided structure, interpolating to t={t_init}...")
        else:
            print(f"Starting from provided structure, noising to t={t_init}...")
        
        num_atoms = int(atom_counts[0].item())
        if int(initial_pos.shape[0]) != num_atoms:
            raise ValueError(
                f"Initial structure has {int(initial_pos.shape[0])} atoms but atom_counts expects {num_atoms}."
            )
        x_0 = initial_pos.to(device).unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, 3)
        noise = center_pos(torch.randn_like(x_0), batch)
        coeff_data, coeff_noise = sampler.scheduler(torch.tensor([t_init], device=device))
        x_t = center_pos(coeff_data * x_0 + coeff_noise * noise, batch)
        start_step = t_init
    else:
        print("Starting from pure random noise...")
        x_t = torch.randn((total_nodes, 3), device=device)
        x_t = center_pos(x_t, batch)
        start_step = T_max - 1

    def capture_positions_for_traj(current_positions: torch.Tensor):
        split = split_by_atom_counts(current_positions.detach().cpu().numpy(), atom_counts_cpu)
        if batch_size == 1:
            return split[0]
        return pad_sample_arrays(split, atom_counts_cpu, pad_value=np.nan)

    def capture_positions_for_npz(current_positions: torch.Tensor):
        split = split_by_atom_counts(current_positions.detach().cpu().numpy(), atom_counts_cpu)
        return pad_sample_arrays(split, atom_counts_cpu, pad_value=np.nan)

    if save_trajectory:
        trajectory = [capture_positions_for_traj(x_t)]
    else:
        trajectory = []

    npz_positions = []
    npz_field_values = []
    npz_tau = []
    npz_alpha_bar = []
    npz_alphas = []
    npz_betas = []
    npz_sqrt_alpha_bar = []
    npz_sqrt_one_minus_alpha_bar = []
    npz_t_prev = []

    # 2. Define the timestep sequence for the reverse process
    if isinstance(sampler, (DDIMSampler, RectifiedFlowSampler, FlowMatchingSampler)):
        if steps is None:
            if isinstance(sampler, DDIMSampler):
                steps = 50
            elif isinstance(sampler, RectifiedFlowSampler):
                steps = 20
            else:
                steps = 20
        time_steps = np.linspace(0, start_step, steps, dtype=int)[::-1].copy()
        print(f"Using {sampler.__class__.__name__} with {len(time_steps)} steps from t={start_step}.")
    else: # DDPMSampler
        time_steps = np.arange(start_step + 1)[::-1].copy()
        print(f"Using DDPM with {len(time_steps)} steps from t={start_step}.")

    # 3. Main sampling loop

    for i, t in enumerate(tqdm(time_steps, desc="Sampling")):
        try:
            # Pre-step check for numerical stability
            if not torch.isfinite(x_t).all():
                print(f"\nWarning: Non-finite coordinates detected at step t={t}. Stopping generation.")
                break

            t_prev = time_steps[i + 1] if i < len(time_steps) - 1 else -1

            edge_index = build_edge_index_from_atom_counts(x_t, atom_counts=atom_counts, cutoff=r_max)
            data = {
                AtomicDataDict.POSITIONS_KEY: x_t,
                AtomicDataDict.NODE_TYPE_KEY: node_types_batch,
                AtomicDataDict.BATCH_KEY: batch,
                AtomicDataDict.EDGE_INDEX_KEY: edge_index,
                AtomicDataDict.T_SAMPLED_KEY: torch.full((batch_size, 1), t, device=device, dtype=torch.long)
            }

            out_dict = model(data)
            model_field_pred = out_dict[field]

            if predict_node_types_field is not None and predict_node_types_field in out_dict:
                node_type_logits = out_dict[predict_node_types_field]
                node_types_batch = torch.argmax(node_type_logits, dim=-1)

            if save_npz:
                npz_positions.append(capture_positions_for_npz(x_t))
                npz_field_values.append(capture_positions_for_npz(model_field_pred))
                npz_t_prev.append(t_prev)
                if isinstance(sampler, FlowMatchingSampler):
                    npz_tau.append(sampler.scheduler.tau[t].item())
                else:
                    npz_alpha_bar.append(sampler.scheduler.alpha_bar[t].item())
                    npz_alphas.append(sampler.scheduler.alphas[t].item())
                    npz_betas.append(sampler.scheduler.betas[t].item())
                    npz_sqrt_alpha_bar.append(sampler.scheduler.sqrt_alpha_bar[t].item())
                    npz_sqrt_one_minus_alpha_bar.append(sampler.scheduler.sqrt_one_minus_alpha_bar[t].item())

            # Call the sampler's step method based on its type
            if isinstance(sampler, DDPMSampler):
                x_t = sampler.step(x_t, t, model_field_pred)
            else:
                if isinstance(sampler, DDIMSampler):
                    x_t = sampler.step(x_t, t, t_prev, model_field_pred, eta=ddim_eta)
                else:
                    x_t = sampler.step(x_t, t, t_prev, model_field_pred)
                
            x_t = center_pos(x_t, batch)

            if save_trajectory:
                trajectory.append(capture_positions_for_traj(x_t))
        except Exception as e:
            print(f"\nAn error occurred during sampling at step t={t}: {e}")
            print("Stopping generation and saving the partial trajectory.")
            break

    print("Sampling complete.")
    npz_data = None
    if save_npz and npz_positions and npz_field_values:
        steps_saved = min(len(npz_positions), len(npz_field_values))
        npz_data = {
            "positions": np.stack(npz_positions[:steps_saved], axis=1),
            "model_field": np.stack(npz_field_values[:steps_saved], axis=1),
            "time_steps": np.array(time_steps[:steps_saved], dtype=int),
            "t_prev": np.array(npz_t_prev[:steps_saved], dtype=int),
            "atom_counts": atom_counts_cpu.numpy(),
        }
        npz_data[field] = npz_data["model_field"]
        if isinstance(sampler, FlowMatchingSampler):
            npz_data["tau"] = np.array(npz_tau[:steps_saved], dtype=np.float32)
        else:
            npz_data["alpha_bar"] = np.array(npz_alpha_bar[:steps_saved], dtype=np.float32)
            npz_data["alphas"] = np.array(npz_alphas[:steps_saved], dtype=np.float32)
            npz_data["betas"] = np.array(npz_betas[:steps_saved], dtype=np.float32)
            npz_data["sqrt_alpha_bar"] = np.array(npz_sqrt_alpha_bar[:steps_saved], dtype=np.float32)
            npz_data["sqrt_one_minus_alpha_bar"] = np.array(npz_sqrt_one_minus_alpha_bar[:steps_saved], dtype=np.float32)
    
    final_positions = split_by_atom_counts(x_t.cpu().numpy(), atom_counts_cpu)
    final_node_types = split_by_atom_counts(node_types_batch.cpu().numpy(), atom_counts_cpu)
    return final_positions, trajectory, npz_data, final_node_types, atom_counts_cpu.numpy()

# --- Script Entry Point (Updated) ---

def main(args=None):
    if args is None:
        args = parse_args()

    # --- 1. Load Model and Initial Structure ---
    
    model, config, _ = CheckpointHandler.load_model(args.model, device=args.device)
    node_types_map, num_types = infer_node_type_metadata(config, args.node_types_map)

    if args.input_structure is not None and (args.min_atoms is not None or args.max_atoms is not None):
        raise ValueError("Use either --input_structure or --min_atoms/--max_atoms, not both.")
    if args.input_structure is None:
        if args.min_atoms is None or args.max_atoms is None:
            raise ValueError("Provide either --input_structure or both --min_atoms and --max_atoms.")
        if args.min_atoms < 1 or args.max_atoms < 1:
            raise ValueError("--min_atoms and --max_atoms must both be >= 1.")
        if args.min_atoms > args.max_atoms:
            raise ValueError("--min_atoms must be <= --max_atoms.")
        if args.predict_node_types_field is None:
            raise ValueError("Sampling without an input structure requires --predict_node_types_field.")
        initial_pos, node_types, elements, atom_group = None, None, None, None
    else:
        initial_pos, node_types, elements, atom_group = load_structure(
            args.input_structure,
            selection=args.selection,
            mol_index=args.mol_index,
            noh=args.noh,
            node_types_map=node_types_map,
        )
        
    if args.predict_node_types_field and node_types_map is None:
        raise ValueError(
            "Could not determine atom labels for predicted node types. Pass --node_types_map or ensure the checkpoint config has type_names."
        )
    if initial_pos is None and num_types is None:
        raise ValueError(
            "Could not determine num_types for random initialization. Pass --node_types_map or ensure the checkpoint config has num_types."
        )
    
    try:
        r_max = config[AtomicDataDict.R_MAX_KEY]
    except KeyError:
        raise KeyError(f"Could not find '{AtomicDataDict.R_MAX_KEY}' in the model's config.")

    # --- 2. Initialize Scheduler and Sampler ---
    
    model_Tmax = infer_model_Tmax(config)
    if args.Tmax is None:
        args.Tmax = model_Tmax if model_Tmax is not None else 1000
    elif model_Tmax is not None and int(args.Tmax) != int(model_Tmax):
        print(
            f"Warning: requested Tmax={args.Tmax} but model was trained with Tmax={model_Tmax}. "
            f"Overriding Tmax to {model_Tmax}."
        )
        args.Tmax = int(model_Tmax)

    if args.sampler == "flow":
        scheduler = FlowMatchingScheduler(T=args.Tmax)
    else:
        scheduler = NoiseScheduler(T=args.Tmax, schedule_type=args.schedule_type)

    if args.sampler == "ddpm":
        sampler = DDPMSampler(scheduler)
    elif args.sampler == "ddim":
        sampler = DDIMSampler(scheduler)
    elif args.sampler == "rectified":
        sampler = RectifiedFlowSampler(scheduler)
    elif args.sampler == "flow":
        sampler = FlowMatchingSampler(scheduler)
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

    schedule_label = args.schedule_type if args.sampler != "flow" else "linear_path"
    final_batches = []
    diffusion_trajectory = None
    npz_positions_batches = []
    npz_field_batches = []
    npz_atom_counts_batches = []
    final_node_types_batches = []
    final_atom_counts_batches = []
    npz_meta = None
    for start_idx in range(0, args.num_samples, batch_size):
        current_batch = min(batch_size, args.num_samples - start_idx)
        if args.num_samples > 1:
            print(f"Running batch {start_idx + 1}-{start_idx + current_batch} of {args.num_samples}...")

        if initial_pos is not None:
            num_atoms = len(initial_pos)
            atom_counts = torch.full((current_batch,), num_atoms, dtype=torch.long)
            init_node_types = node_types.repeat(current_batch)
        else:
            atom_counts = torch.randint(args.min_atoms, args.max_atoms + 1, size=(current_batch,), dtype=torch.long)
            init_node_types = torch.randint(0, num_types, size=(int(atom_counts.sum().item()),), dtype=torch.long)

        final_positions, trajectory, npz_data, final_node_types, atom_counts_np = sample_from_model(
            model=model,
            sampler=sampler,
            device=args.device,
            r_max=r_max,
            atom_counts=atom_counts,
            node_types=init_node_types,
            initial_pos=initial_pos,
            t_init=args.t_init,
            steps=args.steps,
            ddim_eta=args.ddim_eta, # DDIM-specific, ignored by other samplers
            save_trajectory=args.save_trajectory and args.num_samples == 1,
            save_npz=args.save_npz,
            field=args.field,
            predict_node_types_field=args.predict_node_types_field,
        )
        final_batches.extend(final_positions)
        final_node_types_batches.extend(final_node_types)
        final_atom_counts_batches.append(atom_counts_np)

        if args.save_trajectory and args.num_samples == 1:
            diffusion_trajectory = trajectory
        if args.save_npz and npz_data is not None:
            npz_positions_batches.append(npz_data["positions"])
            npz_field_batches.append(npz_data["model_field"])
            npz_atom_counts_batches.append(npz_data["atom_counts"])
            if npz_meta is None:
                npz_meta = {
                    k: v for k, v in npz_data.items()
                    if k not in ("positions", "model_field", "atom_counts", args.field or "noise", "velocity")
                }

    final_positions = final_batches
    final_node_types = final_node_types_batches
    final_atom_counts = np.concatenate(final_atom_counts_batches, axis=0)

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
        file_name_stem = f"generated_molecule_{args.sampler}_{schedule_label}"
        output_file = output_dir / f"{file_name_stem}.xyz"
        traj_base_file = output_dir / f"trajectory_{args.sampler}_{schedule_label}"
        samples_traj_base = output_dir / f"samples_{args.sampler}_{schedule_label}"
    
    with open(output_file, "w") as f:
        for sample_idx, sample_positions in enumerate(final_positions):
            sample_num_atoms = int(final_atom_counts[sample_idx])
            f.write(f"{sample_num_atoms}\n")
            if args.num_samples == 1:
                if args.sampler == "flow":
                    comment = f"Generated by {args.sampler} sampler. Selection: '{args.selection}'"
                else:
                    comment = f"Generated by {args.sampler} sampler with {args.schedule_type} schedule. Selection: '{args.selection}'"
            else:
                if args.sampler == "flow":
                    comment = f"Sample {sample_idx + 1}/{args.num_samples} generated by {args.sampler} sampler. Selection: '{args.selection}'"
                else:
                    comment = (f"Sample {sample_idx + 1}/{args.num_samples} generated by {args.sampler} "
                               f"sampler with {args.schedule_type} schedule. Selection: '{args.selection}'")
            f.write(f"{comment}\n")
            
            if args.predict_node_types_field:
                sample_elements = [node_types_map[int(i)] for i in final_node_types[sample_idx]]
            else:
                sample_elements = elements

            for i, pos in enumerate(sample_positions):
                f.write(f"{sample_elements[i]}   {pos[0]:.4f}   {pos[1]:.4f}   {pos[2]:.4f}\n")
            
    print(f"Successfully saved generated coordinates to: {output_file}")

    if args.save_trajectory:
        if args.num_samples == 1:
            traj_file = traj_base_file.with_suffix(f".{args.traj_format}")
            if args.traj_format == "npz":
                np.savez_compressed(
                    traj_file,
                    trajectory=np.array(diffusion_trajectory),
                    atom_counts=np.array([int(final_atom_counts[0])], dtype=np.int64),
                )
                print(f"Successfully saved trajectory to: {traj_file}")
            else:
                if atom_group is None:
                    raise ValueError("Trajectory export in xtc/dcd format requires --input_structure.")
                save_trajectory_with_mda(str(traj_file), atom_group, diffusion_trajectory)
        else:
            traj_file = samples_traj_base.with_suffix(f".{args.traj_format}")
            if args.traj_format == "npz":
                np.savez_compressed(
                    traj_file,
                    samples=pad_sample_arrays(final_positions, torch.as_tensor(final_atom_counts), pad_value=np.nan),
                    atom_counts=final_atom_counts,
                )
                print(f"Successfully saved samples trajectory to: {traj_file}")
            else:
                if atom_group is None or len(set(final_atom_counts.tolist())) != 1:
                    raise ValueError("Trajectory export in xtc/dcd format requires a fixed input topology for every sample.")
                save_trajectory_with_mda(str(traj_file), atom_group, list(final_positions))

    if args.save_npz and npz_positions_batches and npz_field_batches and npz_atom_counts_batches:
        full_atom_counts = np.concatenate(npz_atom_counts_batches, axis=0)
        global_max_atoms = int(full_atom_counts.max())
        full_positions = np.concatenate(
            [pad_batched_atom_axis(arr, global_max_atoms, pad_value=np.nan) for arr in npz_positions_batches],
            axis=0,
        )
        full_field = np.concatenate(
            [pad_batched_atom_axis(arr, global_max_atoms, pad_value=np.nan) for arr in npz_field_batches],
            axis=0,
        )
        if args.output:
            npz_file = output_base.with_name(output_base.stem + "_full").with_suffix(".npz")
        else:
            npz_file = output_dir / f"full_{args.sampler}_{schedule_label}.npz"
        field_name = args.field
        if field_name is None:
            field_name = "velocity" if args.sampler == "flow" else "noise"
        npz_payload = {
            "positions": full_positions,
            "model_field": full_field,
            "sampler": np.array(args.sampler),
            "schedule_type": np.array(schedule_label),
            "t_init": np.array(-1 if args.t_init is None else args.t_init),
            "steps": np.array(-1 if args.steps is None else args.steps),
            "ddim_eta": np.array(args.ddim_eta),
            "field": np.array(field_name),
            "r_max": np.array(float(r_max)),
            "atom_counts": full_atom_counts,
        }
        npz_payload[field_name] = full_field
        if npz_meta is not None:
            npz_payload.update(npz_meta)
        np.savez_compressed(npz_file, **npz_payload)
        print(f"Successfully saved full sampling data to: {npz_file}")

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="GeqDiff Sampling Script")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to deployed model or .pth model weights.")
    parser.add_argument("-i", "--input_structure", type=str, default=None, help="Path to an input structure file (PDB, GRO, etc.). Optional if --min_atoms/--max_atoms are provided.")
    parser.add_argument("--schedule_type", type=str, default="cosine", choices=["linear", "cosine"], help="Noise schedule to use for diffusion samplers. Ignored by flow matching.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Base path for the output file(s). Suffixes will be added automatically.")
    parser.add_argument("--selection", type=str, default="all", help="MDAnalysis selection string (e.g., 'not name H*').")
    parser.add_argument("--mol_index", type=int, default=0, help="Index of the molecule to use in a multi-molecule file (e.g., SDF). Default is 0.")
    parser.add_argument("--noh", action="store_true", help="Drop hydrogens and shift atom type mapping so C=0.")
    parser.add_argument("--min_atoms", type=int, default=None, help="Minimum number of atoms to sample per molecule when not using --input_structure.")
    parser.add_argument("--max_atoms", type=int, default=None, help="Maximum number of atoms to sample per molecule when not using --input_structure.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for sampling. Defaults to --num_samples.")
    parser.add_argument("--t_init", type=int, default=None, help="Timestep to start denoising from. If not set, starts from pure noise.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for sampling.")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim", "rectified", "flow"], help="Sampler to use.")
    parser.add_argument("-T", "--Tmax", type=int, default=None, help="Maximum number of diffusion timesteps. Defaults to the model's Tmax.")
    parser.add_argument("-f", "--field", type=str, default=None, help="Prediction field to read from the model. Defaults to 'noise' for diffusion samplers and 'velocity' for flow matching.")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps for DDIM, RectifiedFlow, or flow matching sampling.")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="Eta parameter for DDIM sampling (controls stochasticity).")
    parser.add_argument("--save_trajectory", action="store_true", help="Save trajectory. For a single sample, saves the diffusion trajectory. For multiple samples, saves the final samples as a trajectory.")
    parser.add_argument("--save_npz", action="store_true", help="Save per-step positions and predicted noise to an NPZ file.")
    parser.add_argument("--traj_format", type=str, default="dcd", choices=["npz", "xtc", "dcd"], help="Format for the saved trajectory. Default: dcd")
    parser.add_argument("--predict_node_types_field", type=str, default=None, help="If specified, the model will predict node types using the output of this field.")
    parser.add_argument("--node_types_map", type=str, default=None, help="Comma-separated list of atom names for the node type prediction task.")
    
    if arg_list is not None:
        return parser.parse_args(arg_list)
    else:
        return parser.parse_args()

if __name__ == "__main__":
    main()
