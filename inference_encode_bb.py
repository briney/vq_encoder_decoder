import argparse
import datetime
import functools
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list
from box import Box
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import GCPNetDataset, custom_collate_pretrained_gcp
from models.super_model import (
    compile_gcp_encoder,
    compile_non_gcp_and_exclude_vq,
    prepare_model,
)
from utils.utils import get_logging, load_checkpoints_simple, load_configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/inference_encode_config.yaml",
        help="Path to inference YAML config.",
    )
    # Parquet streaming/merge flags (moved from YAML)
    parser.add_argument(
        "--streaming-chunk-size",
        type=int,
        default=10000,
        help="Rows per shard file before flushing to disk.",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=["zstd", "snappy"],
        default="zstd",
        help="Compression codec for Parquet shards.",
    )
    parser.add_argument(
        "--dataset-subdir",
        default="parquet_dataset",
        help="Subdirectory inside the timestamped result_dir to write shards.",
    )
    parser.add_argument(
        "--merge-to-single-parquet",
        action="store_true",
        help="If set, merge shards to a single Parquet file on main process (out-of-core).",
    )
    parser.add_argument(
        "--final-parquet-filename",
        default="vq_indices.parquet",
        help="Filename for the merged Parquet if --merge-to-single-parquet is set.",
    )
    parser.add_argument(
        "--delete-shards-after-merge",
        action="store_true",
        help="If set, delete shard files after successful single-file merge.",
    )
    return parser.parse_args()


def load_saved_encoder_decoder_configs(encoder_cfg_path, decoder_cfg_path):
    # Load encoder and decoder configs from a saved result directory
    with open(encoder_cfg_path) as f:
        enc_cfg = yaml.full_load(f)
    encoder_configs = Box(enc_cfg)

    with open(decoder_cfg_path) as f:
        dec_cfg = yaml.full_load(f)
    decoder_configs = Box(dec_cfg)

    return encoder_configs, decoder_configs


def record_indices(pids, indices_tensor, sequences, records, coords_nested=None):
    """Append pid-index-sequence (and optional coordinates) tuples to records list, ensuring indices is always a list."""
    cpu_inds = indices_tensor.detach().cpu().tolist()
    # Handle scalar to list
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    for i, (pid, idx, seq) in enumerate(zip(pids, cpu_inds, sequences)):
        # wrap non-list idx into list
        if not isinstance(idx, list):
            idx = [idx]
        cleaned = [int(v) for v in idx if v != -1]
        if cleaned:
            cleaned = " ".join(map(str, cleaned))
            rec = {"pid": pid, "indices": cleaned, "protein_sequence": seq}
            if coords_nested is not None:
                rec["coordinates"] = coords_nested[i]
            records.append(rec)
        # records.append({"pid": pid, "indices": cleaned, "protein_sequence": seq})


def reformat_indices(pids, indices_tensor, sequences):
    """Append pid-index-sequence tuples to records list, ensuring indices is always a list."""
    records = []
    cpu_inds = indices_tensor.detach().cpu().tolist()
    # Handle scalar to list
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    for pid, idx, seq in zip(pids, cpu_inds, sequences):
        # wrap non-list idx into list
        if not isinstance(idx, list):
            idx = [idx]
        cleaned = [int(v) for v in idx if v != -1]
        # records.append({"pid": pid, "indices": cleaned, "protein_sequence": seq})
        if cleaned:
            cleaned = " ".join(map(str, cleaned))
            records.append({"pid": pid, "indices": cleaned, "protein_sequence": seq})
    return records


def extract_n_ca_c_coords(batch):
    """
    Return per-sample coordinates (seq_len, 3, 3) in [N, CA, C] order from batch["target_coords"].
    """
    coords_b = batch["target_coords"]  # (B, max_len, 9) flattened [N, CA, C]
    masks_b = batch["masks"]  # (B, max_len) bool
    out = []
    for b in range(coords_b.size(0)):
        L = int(masks_b[b].sum().item())
        if L <= 0:
            out.append([])
            continue
        flat = coords_b[b, :L]  # (L, 9)
        tri = flat.reshape(L, 3, 3)  # (L, 3, 3) [N, CA, C]
        out.append(tri.detach().cpu().tolist())
    return out


def record_indices_typed(pids, indices_tensor, sequences, records, coords_nested=None):
    """
    Append typed records without string-flattening:
    - indices: List[int]
    - coordinates: List[List[List[float]]] in [N, CA, C] order with shape [seq_len, 3, 3]
    """
    cpu_inds = indices_tensor.detach().cpu().tolist()
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    for i, (pid, idx, seq) in enumerate(zip(pids, cpu_inds, sequences)):
        if not isinstance(idx, list):
            idx = [idx]
        cleaned = [int(v) for v in idx if v != -1]
        if cleaned:
            rec = {"pid": pid, "indices": cleaned, "protein_sequence": seq}
            if coords_nested is not None:
                rec["coordinates"] = coords_nested[i]
            records.append(rec)


def make_parquet_schema():
    return pa.schema(
        [
            ("pid", pa.string()),
            ("indices", pa.list_(pa.int32())),
            ("protein_sequence", pa.string()),
            ("coordinates", pa.list_(pa.list_(pa.list_(pa.float32())))),
        ]
    )


def flush_chunk(records_buf, out_dir, rank, chunk_idx, schema, compression="zstd"):
    if not records_buf:
        return chunk_idx
    table = pa.Table.from_pylist(records_buf, schema=schema)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"part-rank{rank:02d}-chunk{chunk_idx:06d}.parquet"
    )
    pq.write_table(table, out_path, compression=compression)
    records_buf.clear()
    return chunk_idx + 1


def main():
    args = parse_args()
    # Load inference configuration
    with open(args.config) as f:
        infer_cfg = yaml.full_load(f)
    infer_cfg = Box(infer_cfg)

    dataloader_config = DataLoaderConfiguration(
        # dispatch_batches=False,
        non_blocking=True,
        even_batches=False,
    )

    # Initialize accelerator for mixed precision and multi-GPU
    accelerator = Accelerator(
        mixed_precision=infer_cfg.mixed_precision, dataloader_config=dataloader_config
    )

    # Setup output directory with timestamp
    now = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

    if accelerator.is_main_process:
        result_dir = os.path.join(infer_cfg.output_base_dir, now)
        os.makedirs(result_dir, exist_ok=True)
        try:
            shutil.copy(args.config, result_dir)
        except Exception:
            pass
        paths = [result_dir]
    else:
        # Initialize with placeholders.
        paths = [None]

    # Broadcast paths to all processes
    broadcast_object_list(paths, from_process=0)
    result_dir = paths[0]

    # Paths to training configs
    vqvae_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_vqvae)
    encoder_cfg_path = os.path.join(
        infer_cfg.trained_model_dir, infer_cfg.config_encoder
    )
    decoder_cfg_path = os.path.join(
        infer_cfg.trained_model_dir, infer_cfg.config_decoder
    )

    # Load main config
    with open(vqvae_cfg_path) as f:
        vqvae_cfg = yaml.full_load(f)
    configs = load_configs(vqvae_cfg)

    # Override task-specific settings
    configs.train_settings.max_task_samples = infer_cfg.get(
        "max_task_samples", configs.train_settings.max_task_samples
    )
    configs.model.max_length = infer_cfg.get("max_length", configs.model.max_length)

    # Load encoder/decoder configs from saved results instead of default utils
    encoder_configs, decoder_configs = load_saved_encoder_decoder_configs(
        encoder_cfg_path, decoder_cfg_path
    )

    # Prepare dataset and dataloader
    dataset = GCPNetDataset(
        infer_cfg.data_path,
        top_k=encoder_configs.top_k,
        num_positional_embeddings=encoder_configs.num_positional_embeddings,
        configs=configs,
        mode="evaluation",
    )
    collate_fn = functools.partial(
        custom_collate_pretrained_gcp,
        featuriser=dataset.pretrained_featuriser,
        task_transform=dataset.pretrained_task_transform,
    )

    loader = DataLoader(
        dataset,
        shuffle=infer_cfg.shuffle,
        batch_size=infer_cfg.batch_size,
        num_workers=infer_cfg.num_workers,
        collate_fn=collate_fn,
    )

    # Setup file logger in result directory
    logger = get_logging(result_dir, configs)

    # Optional: warn if deprecated Parquet keys are found in YAML (ignored in favor of CLI)
    deprecated_parquet_keys = [
        "streaming_chunk_size",
        "parquet_compression",
        "dataset_subdir",
        "merge_to_single_parquet",
        "final_parquet_filename",
        "delete_shards_after_merge",
    ]
    try:
        found_deprecated = [k for k in deprecated_parquet_keys if k in infer_cfg]
        if found_deprecated:
            logger.warning(
                "Deprecated Parquet options in YAML are ignored: "
                + ", ".join(found_deprecated)
                + ". Use CLI flags instead."
            )
    except Exception:
        # If Box membership check fails for any reason, ignore silently.
        pass

    # Prepare model
    model = prepare_model(
        configs,
        logger,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
    )
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(
        infer_cfg.trained_model_dir, infer_cfg.checkpoint_path
    )
    model = load_checkpoints_simple(checkpoint_path, model, logger)

    compile_cfg = infer_cfg.get("compile_model")
    if compile_cfg and compile_cfg.get("enabled", False):
        compile_mode = compile_cfg.get("mode")
        compile_backend = compile_cfg.get("backend", "inductor")
        compile_encoder = compile_cfg.get("compile_encoder", True)

        if (
            compile_encoder
            and hasattr(model, "encoder")
            and getattr(configs.model.encoder, "name", None) == "gcpnet"
        ):
            model = compile_gcp_encoder(
                model, mode=compile_mode, backend=compile_backend
            )
            logger.info("GCP encoder compiled for inference.")

        model = compile_non_gcp_and_exclude_vq(
            model, mode=compile_mode, backend=compile_backend
        )
        logger.info("Compiled VQVAE components for inference (VQ layer excluded).")

    # Prepare everything with accelerator (model and dataloader)
    model, loader = accelerator.prepare(model, loader)

    # Initialize streaming Parquet dataset settings
    rank = accelerator.process_index
    dataset_dir = os.path.join(
        result_dir, getattr(args, "dataset_subdir", "parquet_dataset")
    )
    chunk_size = getattr(args, "streaming_chunk_size", 10_000)
    parquet_compression = getattr(args, "parquet_compression", "zstd")
    schema = make_parquet_schema()
    records_buf = []
    chunk_idx = 0

    # Initialize the progress bar using tqdm (separate from iteration)
    progress_bar = tqdm(
        range(0, int(len(loader))),
        leave=True,
        disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process),
    )
    progress_bar.set_description("Inference")

    for i, batch in enumerate(loader):
        # Inference loop
        with torch.inference_mode():
            # Move graph batch onto accelerator device
            batch["graph"] = batch["graph"].to(accelerator.device)
            batch["masks"] = batch["masks"].to(accelerator.device)
            batch["nan_masks"] = batch["nan_masks"].to(accelerator.device)

            # Forward pass: get either decoded outputs or VQ layer outputs
            output_dict = model(batch, return_vq_layer=True)
            indices = output_dict["indices"]
            pids = batch["pid"]  # list of identifiers
            sequences = batch["seq"]

            # record indices and ground-truth N/CA/C coords per sample
            coords_nested = extract_n_ca_c_coords(batch)
            record_indices_typed(
                pids, indices, sequences, records_buf, coords_nested=coords_nested
            )

            # Flush per-chunk to Parquet
            if len(records_buf) >= chunk_size:
                chunk_idx = flush_chunk(
                    records_buf=records_buf,
                    out_dir=dataset_dir,
                    rank=rank,
                    chunk_idx=chunk_idx,
                    schema=schema,
                    compression=parquet_compression,
                )

            # Update progress bar manually
            progress_bar.update(1)

    # logger.info(f"Inference encoding completed. Results are saved in {result_dir}")
    logger.info("Inference encoding completed.")

    # Flush any remaining buffered records on this rank
    chunk_idx = flush_chunk(
        records_buf=records_buf,
        out_dir=dataset_dir,
        rank=rank,
        chunk_idx=chunk_idx,
        schema=schema,
        compression=parquet_compression,
    )

    # Ensure all processes have completed before any optional merge
    accelerator.wait_for_everyone()

    # Optional: out-of-core merge to a single Parquet on the main process only
    if accelerator.is_main_process and getattr(args, "merge_to_single_parquet", False):
        final_parquet = os.path.join(
            result_dir,
            getattr(args, "final_parquet_filename", "vq_indices.parquet"),
        )
        # Gather all shard files
        try:
            shard_paths = sorted(
                [
                    os.path.join(dataset_dir, f)
                    for f in os.listdir(dataset_dir)
                    if f.endswith(".parquet")
                ]
            )
        except FileNotFoundError:
            shard_paths = []

        if shard_paths:
            with pq.ParquetWriter(
                final_parquet, schema=schema, compression=parquet_compression
            ) as writer:
                for path in shard_paths:
                    pf = pq.ParquetFile(path)
                    for rg in range(pf.num_row_groups):
                        writer.write_table(pf.read_row_group(rg))
            logger.info(f"Single-file Parquet written to {final_parquet}")

            # Optionally delete shards after merge
            if getattr(args, "delete_shards_after_merge", False):
                removed = 0
                for path in shard_paths:
                    try:
                        os.remove(path)
                        removed += 1
                    except OSError:
                        pass
                logger.info(
                    f"Deleted {removed} shard files after merge from {dataset_dir}"
                )
        else:
            logger.warning(
                f"No shard files found in {dataset_dir}; skipping single-file merge."
            )
    else:
        if accelerator.is_main_process:
            logger.info(f"Parquet dataset written to {dataset_dir}")

    # Ensure all processes have completed before exiting
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()


if __name__ == "__main__":
    main()
