#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

STANDARD_SOURCE_DATASET="${ROOT_DIR}/lego/lego_dataset_shell.npz"
STANDARD_DIFFUSION_DATASET="${ROOT_DIR}/lego/lego_diffusion_dataset_shell.npz"
STANDARD_EXPERIMENT_CONFIG="${ROOT_DIR}/config/experiment/lego_flow_matching_shell.yaml"
STANDARD_RESULTS_ROOT="${ROOT_DIR}/results/lego"
STANDARD_SAMPLED_DATASET="${ROOT_DIR}/lego/lego_sampled_shell_dataset.npz"

SMALL_SOURCE_DATASET="${ROOT_DIR}/lego/lego_dataset_shell_small.npz"
SMALL_DIFFUSION_DATASET="${ROOT_DIR}/lego/lego_diffusion_dataset_shell_small.npz"
SMALL_EXPERIMENT_CONFIG="${ROOT_DIR}/config/experiment/lego_flow_matching_shell_small_tiny.yaml"
SMALL_RESULTS_ROOT="${ROOT_DIR}/results/lego"
SMALL_SAMPLED_DATASET="${ROOT_DIR}/lego/lego_sampled_shell_small_dataset.npz"

SOURCE_DATASET_PATH=""
DIFFUSION_DATASET_PATH=""
EXPERIMENT_CONFIG_PATH=""
RESULTS_ROOT_PATH=""
SAMPLED_DATASET_PATH=""
CURRENT_PRESET_LABEL=""
DEFAULT_SOURCE_SAMPLES=""
DEFAULT_MESH_RESOLUTION=""
DEFAULT_OCCUPANCY_MODE=""
DEFAULT_BASE_RADIUS=""
DEFAULT_RADIAL_SCALE=""
DEFAULT_MIN_RADIUS=""
DEFAULT_MAX_RADIUS=""
DEFAULT_SHELL_THICKNESS=""
DEFAULT_SHELL_SPARSITY=""

apply_standard_shell_preset() {
  CURRENT_PRESET_LABEL="standard-shell"
  SOURCE_DATASET_PATH="${STANDARD_SOURCE_DATASET}"
  DIFFUSION_DATASET_PATH="${STANDARD_DIFFUSION_DATASET}"
  EXPERIMENT_CONFIG_PATH="${STANDARD_EXPERIMENT_CONFIG}"
  RESULTS_ROOT_PATH="${STANDARD_RESULTS_ROOT}"
  SAMPLED_DATASET_PATH="${STANDARD_SAMPLED_DATASET}"
  DEFAULT_SOURCE_SAMPLES="100"
  DEFAULT_MESH_RESOLUTION="36"
  DEFAULT_OCCUPANCY_MODE="shell"
  DEFAULT_BASE_RADIUS="5.0"
  DEFAULT_RADIAL_SCALE="1.55"
  DEFAULT_MIN_RADIUS="3.5"
  DEFAULT_MAX_RADIUS="10.5"
  DEFAULT_SHELL_THICKNESS="1.1"
  DEFAULT_SHELL_SPARSITY="0.15"
}

apply_small_shell_preset() {
  CURRENT_PRESET_LABEL="small-shell-overfit"
  SOURCE_DATASET_PATH="${SMALL_SOURCE_DATASET}"
  DIFFUSION_DATASET_PATH="${SMALL_DIFFUSION_DATASET}"
  EXPERIMENT_CONFIG_PATH="${SMALL_EXPERIMENT_CONFIG}"
  RESULTS_ROOT_PATH="${SMALL_RESULTS_ROOT}"
  SAMPLED_DATASET_PATH="${SMALL_SAMPLED_DATASET}"
  DEFAULT_SOURCE_SAMPLES="16"
  DEFAULT_MESH_RESOLUTION="36"
  DEFAULT_OCCUPANCY_MODE="shell"
  DEFAULT_BASE_RADIUS="4.0"
  DEFAULT_RADIAL_SCALE="0.9"
  DEFAULT_MIN_RADIUS="2.75"
  DEFAULT_MAX_RADIUS="7.5"
  DEFAULT_SHELL_THICKNESS="0.9"
  DEFAULT_SHELL_SPARSITY="0.05"
}

apply_standard_shell_preset

find_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s' "${PYTHON_BIN}"
    return
  fi

  local candidate
  for candidate in \
    "${ROOT_DIR}/.venv-geqdiff/bin/python" \
    "${ROOT_DIR}/.venv/bin/python" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return
    fi
  done

  echo "Could not find a Python interpreter. Set PYTHON_BIN or activate a venv first." >&2
  exit 1
}

PYTHON_CMD="$(find_python)"
GEQTRAIN_ROOT="${ROOT_DIR}/deps/GEqTrain"
GEQTRAIN_PYTHONPATH="${GEQTRAIN_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  GEQTRAIN_PYTHONPATH="${GEQTRAIN_PYTHONPATH}:${PYTHONPATH}"
fi

build_train_command() {
  if [[ -f "${GEQTRAIN_ROOT}/geqtrain/scripts/train.py" ]]; then
    TRAIN_CMD=("${PYTHON_CMD}" "${GEQTRAIN_ROOT}/geqtrain/scripts/train.py")
  elif command -v geqtrain-train >/dev/null 2>&1; then
    TRAIN_CMD=(geqtrain-train)
  else
    echo "Could not find GEqTrain training entrypoint." >&2
    exit 1
  fi
}

build_train_command

prompt_with_default() {
  local label="$1"
  local default_value="${2-}"
  local reply=""
  if [[ -n "${default_value}" ]]; then
    read -r -p "${label} [${default_value}]: " reply || true
    printf '%s' "${reply:-${default_value}}"
  else
    read -r -p "${label}: " reply || true
    printf '%s' "${reply}"
  fi
}

prompt_yes_no() {
  local label="$1"
  local default_answer="${2:-y}"
  local suffix="[y/N]"
  local reply=""

  if [[ "${default_answer}" == "y" ]]; then
    suffix="[Y/n]"
  fi

  while true; do
    read -r -p "${label} ${suffix}: " reply || true
    reply="${reply:-${default_answer}}"
    case "${reply,,}" in
      y|yes) return 0 ;;
      n|no) return 1 ;;
      *) echo "Please answer yes or no." ;;
    esac
  done
}

pause_menu() {
  echo
  read -r -p "Press Enter to continue..." _ || true
}

run_cmd() {
  echo
  echo "Running:"
  printf '  %q' "$@"
  echo
  echo
  "$@"
  local status=$?
  echo
  if [[ "${status}" -eq 0 ]]; then
    echo "Command completed successfully."
  else
    echo "Command failed with exit code ${status}."
  fi
  return "${status}"
}

confirm_overwrite() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    return 0
  fi
  prompt_yes_no "Overwrite existing file ${path}?" "n"
}

latest_named_file() {
  local root="$1"
  local name="$2"
  local latest=""
  local file=""

  if [[ ! -d "${root}" ]]; then
    printf '%s' ""
    return
  fi

  while IFS= read -r -d '' file; do
    if [[ -z "${latest}" || "${file}" -nt "${latest}" ]]; then
      latest="${file}"
    fi
  done < <(find "${root}" -type f -name "${name}" -print0 2>/dev/null)

  printf '%s' "${latest}"
}

latest_run_dir() {
  local root="$1"
  local latest=""
  local dir=""

  if [[ ! -d "${root}" ]]; then
    printf '%s' ""
    return
  fi

  while IFS= read -r -d '' dir; do
    if [[ -z "${latest}" || "${dir}" -nt "${latest}" ]]; then
      latest="${dir}"
    fi
  done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

  printf '%s' "${latest}"
}

collect_hydra_overrides() {
  HYDRA_OVERRIDES=()
  echo
  echo "Enter extra Hydra overrides, one per line."
  echo "Examples: max_epochs=1, batch_size=4, +n_train=16, +save_checkpoint_freq=1"
  echo "Leave the prompt blank to finish."
  while true; do
    local override=""
    read -r -p "override> " override || true
    if [[ -z "${override}" ]]; then
      break
    fi
    HYDRA_OVERRIDES+=("${override}")
  done
}

show_defaults() {
  echo
  echo "Current defaults"
  echo "  Preset               ${CURRENT_PRESET_LABEL}"
  echo "  Python               ${PYTHON_CMD}"
  echo "  GEqTrain             ${GEQTRAIN_ROOT}"
  echo "  Source dataset       ${SOURCE_DATASET_PATH}"
  echo "  Diffusion dataset    ${DIFFUSION_DATASET_PATH}"
  echo "  Experiment config    ${EXPERIMENT_CONFIG_PATH}"
  echo "  Results root         ${RESULTS_ROOT_PATH}"
  echo "  Sampled dataset      ${SAMPLED_DATASET_PATH}"
  echo "  Source samples       ${DEFAULT_SOURCE_SAMPLES}"
  echo "  Mesh resolution      ${DEFAULT_MESH_RESOLUTION}"
  echo "  Occupancy mode       ${DEFAULT_OCCUPANCY_MODE}"
  echo "  Base radius          ${DEFAULT_BASE_RADIUS}"
  echo "  Radial scale         ${DEFAULT_RADIAL_SCALE}"
  echo "  Min radius           ${DEFAULT_MIN_RADIUS}"
  echo "  Max radius           ${DEFAULT_MAX_RADIUS}"
  echo "  Shell thickness      ${DEFAULT_SHELL_THICKNESS}"
  echo "  Shell sparsity       ${DEFAULT_SHELL_SPARSITY}"
}

generate_source_dataset() {
  local samples seed output_path mesh_resolution occupancy_mode shell_thickness shell_sparsity
  local base_radius radial_scale min_radius max_radius
  samples="$(prompt_with_default "Number of source LEGO examples" "${DEFAULT_SOURCE_SAMPLES}")"
  seed="$(prompt_with_default "Random seed" "13")"
  output_path="$(prompt_with_default "Output source dataset path" "${SOURCE_DATASET_PATH}")"
  mesh_resolution="$(prompt_with_default "Mesh resolution" "${DEFAULT_MESH_RESOLUTION}")"
  occupancy_mode="$(prompt_with_default "Occupancy mode (solid or shell)" "${DEFAULT_OCCUPANCY_MODE}")"
  base_radius="$(prompt_with_default "Base radius" "${DEFAULT_BASE_RADIUS}")"
  radial_scale="$(prompt_with_default "Radial scale" "${DEFAULT_RADIAL_SCALE}")"
  min_radius="$(prompt_with_default "Min radius" "${DEFAULT_MIN_RADIUS}")"
  max_radius="$(prompt_with_default "Max radius" "${DEFAULT_MAX_RADIUS}")"
  shell_thickness="${DEFAULT_SHELL_THICKNESS}"
  shell_sparsity="${DEFAULT_SHELL_SPARSITY}"
  if [[ "${occupancy_mode}" == "shell" ]]; then
    shell_thickness="$(prompt_with_default "Shell thickness" "${DEFAULT_SHELL_THICKNESS}")"
    shell_sparsity="$(prompt_with_default "Shell sparsity [0-0.95]" "${DEFAULT_SHELL_SPARSITY}")"
  fi

  if ! confirm_overwrite "${output_path}"; then
    echo "Cancelled."
    return
  fi

  mkdir -p "$(dirname "${output_path}")"
  if run_cmd "${PYTHON_CMD}" "${ROOT_DIR}/lego/lego_engine.py" \
    --samples "${samples}" \
    --seed "${seed}" \
    --path "${output_path}" \
    --mesh-resolution "${mesh_resolution}" \
    --base-radius "${base_radius}" \
    --radial-scale "${radial_scale}" \
    --min-radius "${min_radius}" \
    --max-radius "${max_radius}" \
    --occupancy-mode "${occupancy_mode}" \
    --shell-thickness "${shell_thickness}" \
    --shell-sparsity "${shell_sparsity}"; then
    SOURCE_DATASET_PATH="${output_path}"
  fi
}

generate_diffusion_dataset() {
  local input_path output_path splits strategy ligand_size ligand_min ligand_max ligand_fraction seed
  input_path="$(prompt_with_default "Input source dataset path" "${SOURCE_DATASET_PATH}")"
  output_path="$(prompt_with_default "Output diffusion dataset path" "${DIFFUSION_DATASET_PATH}")"
  splits="$(prompt_with_default "Splits per frame" "4")"
  strategy="$(prompt_with_default "Split strategy (connected or radius)" "connected")"
  ligand_size="$(prompt_with_default "Exact ligand size (blank to use min/max)" "")"
  ligand_min="$(prompt_with_default "Ligand size min" "2")"
  ligand_max="$(prompt_with_default "Ligand size max" "8")"
  ligand_fraction="$(prompt_with_default "Ligand fraction (blank to disable)" "")"
  seed="$(prompt_with_default "Random seed" "17")"

  if ! confirm_overwrite "${output_path}"; then
    echo "Cancelled."
    return
  fi

  mkdir -p "$(dirname "${output_path}")"
  local cmd=(
    "${PYTHON_CMD}" "${ROOT_DIR}/geqdiff/scripts/build_diffusion_dataset.py"
    --input "${input_path}"
    --output "${output_path}"
    --splits-per-frame "${splits}"
    --split-strategy "${strategy}"
    --seed "${seed}"
  )

  if [[ -n "${ligand_size}" ]]; then
    cmd+=(--ligand-size "${ligand_size}")
  else
    cmd+=(--ligand-size-min "${ligand_min}" --ligand-size-max "${ligand_max}")
    if [[ -n "${ligand_fraction}" ]]; then
      cmd+=(--ligand-fraction "${ligand_fraction}")
    fi
  fi

  if run_cmd "${cmd[@]}"; then
    DIFFUSION_DATASET_PATH="${output_path}"
  fi
}

inspect_diffusion_dataset() {
  local input_path index
  input_path="$(prompt_with_default "Diffusion dataset path to inspect" "${DIFFUSION_DATASET_PATH}")"
  index="$(prompt_with_default "Detailed example index (blank for summary only)" "")"

  local cmd=("${PYTHON_CMD}" "${ROOT_DIR}/geqdiff/scripts/inspect_diffusion_dataset.py" --input "${input_path}")
  if [[ -n "${index}" ]]; then
    cmd+=(--index "${index}")
  fi

  run_cmd "${cmd[@]}"
}

train_lego_model() {
  local config_path dataset_path results_root device run_name
  config_path="$(prompt_with_default "Experiment config path" "${EXPERIMENT_CONFIG_PATH}")"
  device="$(prompt_with_default "Device" "cuda:0")"

  collect_hydra_overrides

  local cmd=(env
    "PYTHONPATH=${GEQTRAIN_PYTHONPATH}"
    "${TRAIN_CMD[@]}" "${config_path}" -d "${device}"
  )
  local override
  for override in "${HYDRA_OVERRIDES[@]}"; do
    cmd+=(-o "${override}")
  done

  if run_cmd "${cmd[@]}"; then
    EXPERIMENT_CONFIG_PATH="${config_path}"
    RESULTS_ROOT_PATH="${results_root}"
  fi
}

sample_lego_blocks() {
  local latest_model model_path input_path output_path source_path metrics_json num_samples steps sampler_name late_refine_from_step late_refine_factor linger_step linger_count clash_guidance clash_guidance_strength clash_guidance_max_norm clash_guidance_weight_schedule clash_guidance_auto_scale clash_guidance_auto_scale_min clash_guidance_auto_scale_max cohesion_guidance_strength cohesion_guidance_target_contacts save_metrics device seed indices_raw save_intermediates use_refinement use_linger use_guidance use_cohesion
  latest_model="$(latest_named_file "${RESULTS_ROOT_PATH}" "best_model.pth")"
  model_path="$(prompt_with_default "Checkpoint path" "${latest_model}")"
  input_path="$(prompt_with_default "Input diffusion dataset path" "${DIFFUSION_DATASET_PATH}")"
  output_path="$(prompt_with_default "Output sampled LEGO dataset path" "${SAMPLED_DATASET_PATH}")"
  metrics_json="$(prompt_with_default "Metrics JSON path (blank for auto next to NPZ)" "")"
  source_path="$(prompt_with_default "Canonical source dataset path (blank to skip enrichment)" "")"
  num_samples="$(prompt_with_default "Number of samples to draw" "4")"
  steps="$(prompt_with_default "Reverse integration steps" "20")"
  sampler_name="$(prompt_with_default "Flow-matching sampler (heun/euler)" "heun")"

  if prompt_yes_no "Enable late refinement (substeps near low tau)?" "n"; then
    use_refinement="y"
    late_refine_from_step="$(prompt_with_default "Late refine from discrete step" "3")"
    late_refine_factor="$(prompt_with_default "Late refine factor" "2")"
  else
    use_refinement="n"
    late_refine_from_step="-1"
    late_refine_factor="1"
  fi

  if prompt_yes_no "Enable experimental linger micro-steps?" "n"; then
    use_linger="y"
    linger_step="$(prompt_with_default "Experimental linger step" "1")"
    linger_count="$(prompt_with_default "Experimental linger count" "3")"
  else
    use_linger="n"
    linger_step="-1"
    linger_count="0"
  fi

  if prompt_yes_no "Enable clash guidance?" "n"; then
    use_guidance="y"
    clash_guidance="y"
    clash_guidance_strength="$(prompt_with_default "Clash guidance strength" "0.05")"
    clash_guidance_max_norm="$(prompt_with_default "Clash guidance max norm" "1.0")"
    clash_guidance_weight_schedule="$(prompt_with_default "Clash guidance weight schedule" "late_quadratic")"
    if prompt_yes_no "Auto-scale clash guidance to model velocity?" "n"; then
      clash_guidance_auto_scale="y"
      clash_guidance_auto_scale_min="$(prompt_with_default "Clash guidance auto-scale min" "0.2")"
      clash_guidance_auto_scale_max="$(prompt_with_default "Clash guidance auto-scale max" "5.0")"
    else
      clash_guidance_auto_scale="n"
      clash_guidance_auto_scale_min="0.2"
      clash_guidance_auto_scale_max="5.0"
    fi
  else
    use_guidance="n"
    clash_guidance="n"
    clash_guidance_strength="0.05"
    clash_guidance_max_norm="1.0"
    clash_guidance_weight_schedule="late_quadratic"
    clash_guidance_auto_scale="n"
    clash_guidance_auto_scale_min="0.2"
    clash_guidance_auto_scale_max="5.0"
  fi

  if prompt_yes_no "Enable cohesion guidance term?" "n"; then
    use_cohesion="y"
    cohesion_guidance_strength="$(prompt_with_default "Cohesion guidance strength" "0.02")"
    cohesion_guidance_target_contacts="$(prompt_with_default "Cohesion target contacts" "1.5")"
  else
    use_cohesion="n"
    cohesion_guidance_strength="0.0"
    cohesion_guidance_target_contacts="1.5"
  fi
  if [[ "${use_guidance}" != "y" ]]; then
    use_cohesion="n"
    cohesion_guidance_strength="0.0"
    cohesion_guidance_target_contacts="1.5"
  fi

  device="$(prompt_with_default "Device" "cuda:0")"
  seed="$(prompt_with_default "Random seed" "0")"
  indices_raw="$(prompt_with_default "Explicit diffusion example indices (space-separated, blank for random)" "")"
  if prompt_yes_no "Save all intermediate reverse states?" "y"; then
    save_intermediates="y"
  else
    save_intermediates="n"
  fi
  if prompt_yes_no "Save evaluation metrics JSON?" "y"; then
    save_metrics="y"
  else
    save_metrics="n"
  fi

  if ! confirm_overwrite "${output_path}"; then
    echo "Cancelled."
    return
  fi

  mkdir -p "$(dirname "${output_path}")"
  local cmd=(env "PYTHONPATH=${GEQTRAIN_PYTHONPATH}"
    "${PYTHON_CMD}" "${ROOT_DIR}/geqdiff/scripts/sample_lego.py"
    --model "${model_path}"
    --input "${input_path}"
    --output "${output_path}"
    --num-samples "${num_samples}"
    --steps "${steps}"
    --sampler "${sampler_name}"
    --device "${device}"
    --seed "${seed}"
  )
  if [[ "${use_refinement}" == "y" ]]; then
    cmd+=(--late-refine-from-step "${late_refine_from_step}" --late-refine-factor "${late_refine_factor}")
  fi
  if [[ "${use_linger}" == "y" ]]; then
    cmd+=(--linger-step "${linger_step}" --linger-count "${linger_count}")
  fi
  if [[ "${clash_guidance}" == "y" ]]; then
    cmd+=(--clash-guidance --clash-guidance-strength "${clash_guidance_strength}" --clash-guidance-max-norm "${clash_guidance_max_norm}" --clash-guidance-weight-schedule "${clash_guidance_weight_schedule}")
  else
    cmd+=(--no-clash-guidance)
  fi
  if [[ "${clash_guidance_auto_scale}" == "y" ]]; then
    cmd+=(--clash-guidance-auto-scale --clash-guidance-auto-scale-min "${clash_guidance_auto_scale_min}" --clash-guidance-auto-scale-max "${clash_guidance_auto_scale_max}")
  else
    cmd+=(--no-clash-guidance-auto-scale)
  fi
  if [[ "${use_cohesion}" == "y" ]]; then
    cmd+=(--cohesion-guidance-strength "${cohesion_guidance_strength}" --cohesion-guidance-target-contacts "${cohesion_guidance_target_contacts}")
  fi
  if [[ -n "${source_path}" ]]; then
    cmd+=(--source-canonical "${source_path}")
  fi
  if [[ -n "${indices_raw}" ]]; then
    local indices=()
    read -r -a indices <<< "${indices_raw}"
    cmd+=(--indices "${indices[@]}")
  fi
  if [[ "${save_intermediates}" == "y" ]]; then
    cmd+=(--save-intermediates)
  fi
  if [[ "${save_metrics}" == "y" ]]; then
    cmd+=(--save-metrics)
  else
    cmd+=(--no-save-metrics)
  fi
  if [[ -n "${metrics_json}" ]]; then
    cmd+=(--metrics-json "${metrics_json}")
  fi

  if run_cmd "${cmd[@]}"; then
    SAMPLED_DATASET_PATH="${output_path}"
  fi
}

visualize_lego_dataset() {
  local default_dataset dataset_path output_html trajectory_stride
  default_dataset="${SAMPLED_DATASET_PATH}"
  if [[ ! -f "${default_dataset}" ]]; then
    default_dataset="${SOURCE_DATASET_PATH}"
  fi

  dataset_path="$(prompt_with_default "Dataset path to visualize" "${default_dataset}")"
  output_html="$(prompt_with_default "Optional HTML save path (blank to open only)" "")"
  trajectory_stride="$(prompt_with_default "Trajectory frame stride" "1")"

  local cmd=(
    "${PYTHON_CMD}" "${ROOT_DIR}/lego/lego_visualizer.py"
    --path "${dataset_path}"
    --show-dipoles
    --trajectory-stride "${trajectory_stride}"
  )
  if [[ -n "${output_html}" ]]; then
    cmd+=(--output-html "${output_html}")
  fi

  run_cmd "${cmd[@]}"
}

tail_training_log() {
  local run_root run_dir lines log_path
  run_root="$(prompt_with_default "Results root directory" "${RESULTS_ROOT_PATH}")"
  run_dir="$(latest_run_dir "${run_root}")"
  run_dir="$(prompt_with_default "Run directory to inspect" "${run_dir}")"
  lines="$(prompt_with_default "Number of log lines" "40")"
  log_path="${run_dir%/}/log"

  if [[ ! -f "${log_path}" ]]; then
    echo
    echo "No log file found at ${log_path}"
    return
  fi

  run_cmd tail -n "${lines}" "${log_path}"
}

print_menu() {
  echo
  echo "LEGO Workflow Menu"
  echo "  1) Generate source LEGO dataset"
  echo "  2) Generate LEGO diffusion dataset"
  echo "  3) Inspect LEGO diffusion dataset"
  echo "  4) Train LEGO flow-matching model"
  echo "  5) Sample masked LEGO assemblies"
  echo "  6) Visualize LEGO dataset"
  echo "  7) Tail latest training log"
  echo "  8) Show current defaults"
  echo "  9) Use standard shell preset"
  echo " 10) Use small-shell overfit preset"
  echo "  0) Exit"
}

main() {
  while true; do
    print_menu
    local choice
    read -r -p "Choose an option: " choice || true
    case "${choice}" in
      1) generate_source_dataset; pause_menu ;;
      2) generate_diffusion_dataset; pause_menu ;;
      3) inspect_diffusion_dataset; pause_menu ;;
      4) train_lego_model; pause_menu ;;
      5) sample_lego_blocks; pause_menu ;;
      6) visualize_lego_dataset; pause_menu ;;
      7) tail_training_log; pause_menu ;;
      8) show_defaults; pause_menu ;;
      9) apply_standard_shell_preset; show_defaults; pause_menu ;;
      10) apply_small_shell_preset; show_defaults; pause_menu ;;
      0) exit 0 ;;
      *) echo "Unknown option: ${choice}"; pause_menu ;;
    esac
  done
}

main "$@"
