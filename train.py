import argparse
import os
import time
from datetime import timedelta

import numpy as np
import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from tqdm import tqdm

from data.dataset import prepare_gcpnet_vqvae_dataloaders
from models.super_model import (
    compile_gcp_encoder,
    compile_non_gcp_and_exclude_vq,
    prepare_model,
)
from utils.custom_losses import calculate_decoder_loss, log_per_loss_grad_norms
from utils.training_helpers import (
    accumulate_losses,
    average_losses,
    compute_activation,
    compute_metrics,
    finalize_step,
    init_accumulator,
    init_metrics,
    log_tensorboard_epoch,
    log_wandb_epoch,
    progress_postfix,
    reset_metrics,
    update_metrics,
    update_unique_indices,
)
from utils.utils import (
    get_logging,
    load_checkpoints,
    load_configs,
    load_encoder_decoder_configs,
    prepare_optimizer,
    prepare_saving_dir,
    prepare_tensorboard,
    save_backbone_pdb,
    save_checkpoint,
)


def train_loop(net, train_loader, epoch, adaptive_loss_coeffs, **kwargs):
    accelerator = kwargs.pop("accelerator")
    optimizer = kwargs.pop("optimizer")
    scheduler = kwargs.pop("scheduler")
    configs = kwargs.pop("configs")
    writer = kwargs.pop("writer")
    logging = kwargs.pop("logging")
    profiler = kwargs.pop("profiler")
    profile_train_loop = kwargs.pop("profile_train_loop")
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    accum_iter = configs.train_settings.grad_accumulation
    alignment_strategy = configs.train_settings.losses.alignment_strategy

    # Initialize metrics and accumulators
    metrics = init_metrics(configs, accelerator)
    acc = init_accumulator(accum_iter)

    optimizer.zero_grad()

    global_step = kwargs.get("global_step", 0)
    max_steps = kwargs.get("max_steps", float("inf"))
    steps_per_epoch = kwargs.get("steps_per_epoch", 1)

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(
        range(0, int(np.ceil(len(train_loader) / accum_iter))),
        leave=False,
        disable=not (configs.tqdm_progress_bar and accelerator.is_main_process),
    )
    progress_bar.set_description(f"Epoch {epoch}")

    net.train()
    for i, data in enumerate(train_loader):
        with accelerator.accumulate(net):
            if profile_train_loop:
                profiler.step()
                if i >= 1000:  # Profile only the first 1000 steps
                    logging.info("Profiler finished, exiting train step loop.")
                    break

            masks = torch.logical_and(data["masks"], data["nan_masks"])

            optimizer.zero_grad()
            output_dict = net(data)

            # Compute the loss components (function unwraps tensors internally)
            loss_dict, trans_pred_coords, trans_true_coords = calculate_decoder_loss(
                output_dict=output_dict,
                data=data,
                configs=configs,
                alignment_strategy=alignment_strategy,
                adaptive_loss_coeffs=adaptive_loss_coeffs,
            )

            # Apply sample weights to loss if enabled
            if configs.train_settings.sample_weighting.enabled:
                sample_weights = data["sample_weights"]
                # Use the mean sample weight for the batch (could be weighted by batch size)
                batch_weight = sample_weights.mean()
                loss_dict["rec_loss"] = loss_dict["rec_loss"] * batch_weight

            # Log per-loss gradient norms and adjust adaptive coefficients
            adaptive_loss_coeffs = log_per_loss_grad_norms(
                loss_dict,
                net,
                configs,
                writer,
                accelerator,
                global_step,
                adaptive_loss_coeffs,
            )

            if (
                accelerator.is_main_process
                and int(epoch) % configs.train_settings.save_pdb_every == 0
                and int(epoch) != 0
                and i == 0
            ):
                logging.info(
                    f"Building PDB files for training data in epoch {epoch:.2f}"
                )
                save_backbone_pdb(
                    trans_pred_coords.detach(),
                    masks,
                    data["pid"],
                    os.path.join(
                        kwargs["result_path"],
                        "pdb_files",
                        f"train_outputs_epoch_{int(epoch)}_step_{i + 1}",
                    ),
                )
                save_backbone_pdb(
                    trans_true_coords.detach().squeeze(),
                    masks,
                    data["pid"],
                    os.path.join(
                        kwargs["result_path"],
                        "pdb_files",
                        f"train_labels_epoch_{int(epoch)}_step_{i + 1}",
                    ),
                )
                logging.info("PDB files are built")

            # Update metrics and accumulators
            update_metrics(
                metrics,
                trans_pred_coords,
                trans_true_coords,
                masks,
                output_dict,
                ignore_index=-100,
            )
            accumulate_losses(
                acc, loss_dict, output_dict, configs, accelerator, use_output_vq=False
            )
            update_unique_indices(acc, output_dict["indices"], accelerator)

            accelerator.backward(loss_dict["step_loss"])
            if accelerator.sync_gradients:
                if (
                    global_step % configs.train_settings.gradient_norm_logging_freq == 0
                    and global_step > 0
                ):
                    # Calculate the gradient norm every configs.train_settings.gradient_norm_logging_freq steps
                    grad_norm = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), 2)
                                for p in net.parameters()
                                if p.grad is not None and p.requires_grad
                            ]
                        ),
                        2,
                    )
                if accelerator.is_main_process and configs.tensorboard_log:
                    writer.add_scalar(
                        "gradient norm/total_amp_scaled",
                        grad_norm.item(),
                        global_step,
                    )

                # Log gradient norm to wandb
                if configs.wandb.enabled:
                    accelerator.log(
                        {"train/gradient_norm/total_amp_scaled": grad_norm.item()},
                        step=global_step,
                    )

                # Accelerate Gradient clipping: unscale the gradients (only when using FP16 AMP) and then apply clipping
                accelerator.clip_grad_norm_(
                    net.parameters(), configs.optimizer.grad_clip_norm
                )

                optimizer.step()
                scheduler.step()

                progress_bar.update(1)
                global_step += 1

                finalize_step(acc)

                if accelerator.is_main_process and configs.tensorboard_log:
                    writer.add_scalar(
                        "lr", optimizer.param_groups[0]["lr"], global_step
                    )

                # Log learning rate to wandb
                if configs.wandb.enabled:
                    accelerator.log(
                        {"train/lr": optimizer.param_groups[0]["lr"]},
                        step=global_step,
                    )

                avgs = average_losses(acc)
                # Calculate fractional epoch for display
                step_in_epoch = acc["counter"]
                fractional_epoch = epoch - 1 + (step_in_epoch / steps_per_epoch)
                progress_bar.set_description(
                    f"epoch {fractional_epoch:.2f} "
                    + f"[loss: {avgs['avg_unscaled_step_loss']:.3f}, "
                    + f"rec loss: {avgs['avg_unscaled_rec_loss']:.3f}, "
                    + f"vq loss: {avgs['avg_unscaled_vq_loss']:.3f}]"
                )

                # Check if we've reached max_steps
                if global_step >= max_steps:
                    logging.info(
                        f"Reached max_steps={max_steps}, stopping training early"
                    )
                    break

            progress_bar.set_postfix(
                progress_postfix(optimizer, loss_dict, global_step)
            )

    # Compute average losses and metrics
    avgs = average_losses(acc)
    metrics_values = compute_metrics(metrics)
    avg_activation = compute_activation(acc, codebook_size)

    # Log metrics to TensorBoard
    if accelerator.is_main_process and configs.tensorboard_log:
        include_ntp = (
            getattr(configs.train_settings.losses, "next_token_prediction", None)
            and configs.train_settings.losses.next_token_prediction.enabled
        )
        log_tensorboard_epoch(
            writer,
            avgs,
            metrics_values,
            epoch,
            activation_percent=np.round(avg_activation * 100, 1),
            include_ntp=include_ntp,
            global_step=global_step,
        )

    # Log metrics to wandb
    if configs.wandb.enabled:
        include_ntp = (
            getattr(configs.train_settings.losses, "next_token_prediction", None)
            and configs.train_settings.losses.next_token_prediction.enabled
        )
        log_wandb_epoch(
            accelerator,
            "train",
            avgs,
            metrics_values,
            epoch,
            activation_percent=np.round(avg_activation * 100, 1),
            include_ntp=include_ntp,
            global_step=global_step,
        )

    # Reset the metrics for the next epoch
    reset_metrics(metrics)

    # Calculate fractional epoch at end of training loop
    step_in_epoch = acc["counter"]
    fractional_epoch = epoch - 1 + (step_in_epoch / steps_per_epoch)
    early_stop = global_step >= max_steps

    return_dict = {
        "loss": avgs["avg_unscaled_step_loss"],
        "rec_loss": avgs["avg_unscaled_rec_loss"],
        "ntp_loss": avgs["avg_unscaled_ntp_loss"],
        "vq_loss": avgs["avg_unscaled_vq_loss"],
        "mae": metrics_values["mae"],
        "rmsd": metrics_values["rmsd"],
        "gdtts": metrics_values["gdtts"],
        "tm_score": metrics_values["tm_score"],
        "perplexity": metrics_values["perplexity"],
        "padding_accuracy": metrics_values.get(
            "tik_tok_padding_accuracy", float("nan")
        ),
        "activation": np.round(avg_activation * 100, 1),
        "counter": acc["counter"],
        "global_step": global_step,
        "adaptive_loss_coeffs": adaptive_loss_coeffs,
        "early_stop": early_stop,
        "fractional_epoch": fractional_epoch,
    }
    return return_dict


def valid_loop(net, valid_loader, epoch, **kwargs):
    optimizer = kwargs.pop("optimizer")
    configs = kwargs.pop("configs")
    accelerator = kwargs.pop("accelerator")
    writer = kwargs.pop("writer")
    logging = kwargs.pop("logging")
    global_step = kwargs.get("global_step", 0)
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    alignment_strategy = configs.train_settings.losses.alignment_strategy

    # Initialize metrics and accumulators for validation
    metrics = init_metrics(configs, accelerator)
    acc = init_accumulator(accum_iter=1)

    optimizer.zero_grad()

    # Initialize the progress bar using tqdm (epoch can be fractional now)
    progress_bar = tqdm(
        range(0, int(len(valid_loader))),
        leave=False,
        disable=not (configs.tqdm_progress_bar and accelerator.is_main_process),
    )
    progress_bar.set_description(f"Validation epoch {epoch:.2f}")

    net.eval()
    for i, data in enumerate(valid_loader):
        with torch.inference_mode():
            masks = torch.logical_and(data["masks"], data["nan_masks"])

            output_dict = net(data)

            update_unique_indices(acc, output_dict["indices"], accelerator)

            # Compute the loss components using dict-style outputs like train loop
            loss_dict, trans_pred_coords, trans_true_coords = calculate_decoder_loss(
                output_dict=output_dict,
                data=data,
                configs=configs,
                alignment_strategy=alignment_strategy,
                adaptive_loss_coeffs=None,
            )

            if (
                accelerator.is_main_process
                and int(epoch) % configs.valid_settings.save_pdb_every == 0
                and int(epoch) != 0
                and i == 0
            ):
                logging.info(
                    f"Building PDB files for validation data in epoch {epoch:.2f}"
                )
                save_backbone_pdb(
                    trans_pred_coords.detach(),
                    masks,
                    data["pid"],
                    os.path.join(
                        kwargs["result_path"],
                        "pdb_files",
                        f"valid_outputs_epoch_{int(epoch)}_step_{i + 1}",
                    ),
                )
                save_backbone_pdb(
                    trans_true_coords.detach(),
                    masks,
                    data["pid"],
                    os.path.join(
                        kwargs["result_path"],
                        "pdb_files",
                        f"valid_labels_epoch_{int(epoch)}_step_{i + 1}",
                    ),
                )
                logging.info("PDB files are built")

            # Update metrics and losses
            update_metrics(
                metrics,
                trans_pred_coords,
                trans_true_coords,
                masks,
                output_dict,
                ignore_index=-100,
            )
            accumulate_losses(
                acc, loss_dict, output_dict, configs, accelerator, use_output_vq=True
            )
            # Finalize this validation step so totals/averages are updated
            finalize_step(acc)

        progress_bar.update(1)
        avgs = average_losses(acc)
        progress_bar.set_description(
            f"validation epoch {epoch:.2f} "
            + f"[loss: {avgs['avg_unscaled_step_loss']:.3f}, "
            + f"rec loss: {avgs['avg_unscaled_rec_loss']:.3f}, "
            + f"vq loss: {avgs['avg_unscaled_vq_loss']:.3f}]"
        )

    # Compute averages and metrics
    avgs = average_losses(acc)
    avg_activation = compute_activation(acc, codebook_size)
    metrics_values = compute_metrics(metrics)

    # Log metrics to TensorBoard
    if accelerator.is_main_process and configs.tensorboard_log:
        include_ntp = (
            getattr(configs.train_settings.losses, "next_token_prediction", None)
            and configs.train_settings.losses.next_token_prediction.enabled
        )
        log_tensorboard_epoch(
            writer,
            avgs,
            metrics_values,
            epoch,
            activation_percent=np.round(avg_activation * 100, 1),
            include_ntp=include_ntp,
            global_step=global_step,
        )

    # Log metrics to wandb
    if configs.wandb.enabled:
        include_ntp = (
            getattr(configs.train_settings.losses, "next_token_prediction", None)
            and configs.train_settings.losses.next_token_prediction.enabled
        )
        log_wandb_epoch(
            accelerator,
            "eval",
            avgs,
            metrics_values,
            epoch,
            activation_percent=np.round(avg_activation * 100, 1),
            include_ntp=include_ntp,
            global_step=global_step,
        )

    # Reset metrics for the next epoch
    reset_metrics(metrics)

    return_dict = {
        "loss": avgs["avg_unscaled_step_loss"],
        "rec_loss": avgs["avg_unscaled_rec_loss"],
        "vq_loss": avgs["avg_unscaled_vq_loss"],
        "ntp_loss": avgs["avg_unscaled_ntp_loss"],
        "mae": metrics_values["mae"],
        "rmsd": metrics_values["rmsd"],
        "gdtts": metrics_values["gdtts"],
        "tm_score": metrics_values["tm_score"],
        "perplexity": metrics_values["perplexity"],
        "padding_accuracy": metrics_values.get(
            "tik_tok_padding_accuracy", float("nan")
        ),
        "activation": np.round(avg_activation * 100, 1),
        "counter": acc["counter"],
    }
    return return_dict


def main(dict_config, config_file_path):
    configs = load_configs(dict_config)
    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    # Set find_unused_parameters to True
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=configs.find_unused_parameters
    )
    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=configs.dispatch_batches,
        even_batches=configs.even_batches,
        non_blocking=configs.non_blocking,
        split_batches=configs.split_batches,
        # use_stateful_dataloader=True
    )
    accelerator = Accelerator(
        kwargs_handlers=[
            ddp_kwargs,
            InitProcessGroupKwargs(timeout=timedelta(minutes=20)),
        ],
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dataloader_config=dataloader_config,
        log_with="wandb" if configs.wandb.enabled else None,
    )

    # Initialize paths to avoid unassigned variable warnings
    result_path, checkpoint_path = None, None

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
        paths = [result_path, checkpoint_path]
    else:
        # Initialize with placeholders.
        paths = [None, None]

    if accelerator.num_processes > 1:
        import torch.distributed as dist

        # Broadcast the list of strings from the main process (src=0) to all others.
        dist.broadcast_object_list(paths, src=0)

        # Now every process has the shared values.
        result_path, checkpoint_path = paths

    encoder_configs, decoder_configs = load_encoder_decoder_configs(
        configs, result_path
    )

    logging = get_logging(result_path, configs)

    # Initialize wandb tracker if enabled
    if configs.wandb.enabled:
        # Build wandb init kwargs
        wandb_init_kwargs = {
            "wandb": {
                "name": configs.wandb.name,
                "tags": configs.wandb.tags if configs.wandb.tags else None,
                "notes": configs.wandb.notes if configs.wandb.notes else None,
                "group": configs.wandb.group,
            }
        }
        # Add entity if specified
        if configs.wandb.entity:
            wandb_init_kwargs["wandb"]["entity"] = configs.wandb.entity

        # Remove None values to use wandb defaults
        wandb_init_kwargs["wandb"] = {
            k: v for k, v in wandb_init_kwargs["wandb"].items() if v is not None
        }

        accelerator.init_trackers(
            project_name=configs.wandb.project,
            config=dict_config,
            init_kwargs=wandb_init_kwargs if wandb_init_kwargs["wandb"] else None,
        )
        logging.info(f"Initialized wandb tracker for project: {configs.wandb.project}")

    train_dataloader, valid_dataloader = prepare_gcpnet_vqvae_dataloaders(
        logging,
        accelerator,
        configs,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
    )
    logging.info("preparing dataloaders are done")

    net = prepare_model(
        configs,
        logging,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
    )
    logging.info("preparing models is done")

    optimizer, scheduler = prepare_optimizer(
        net, configs, len(train_dataloader), logging
    )
    logging.info("preparing optimizer is done")

    net, start_epoch, start_global_step = load_checkpoints(
        configs, optimizer, scheduler, logging, net, accelerator
    )

    # compile models to train faster and efficiently
    if configs.model.compile_model:
        if hasattr(net, "encoder") and configs.model.encoder.name == "gcpnet":
            net = compile_gcp_encoder(net, mode=None, backend="inductor")
            logging.info("GCP encoder compiled.")
        net = compile_non_gcp_and_exclude_vq(net, mode=None, backend="inductor")
        logging.info("All GCP-VQVAE layers compiled except VQ layer.")
    net, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, valid_dataloader, scheduler
    )

    net.to(accelerator.device)

    if accelerator.is_main_process:
        # initialize tensorboards
        train_writer, valid_writer = prepare_tensorboard(result_path)
    else:
        train_writer, valid_writer = None, None

    # Calculate steps per epoch and max steps
    steps_per_epoch = int(
        np.ceil(len(train_dataloader) / configs.train_settings.grad_accumulation)
    )

    if accelerator.is_main_process:
        logging.info(f"number of train steps per epoch: {steps_per_epoch}")

    # Determine max steps and checkpoint/validation intervals
    if getattr(configs.train_settings, "max_train_steps", None) is not None:
        max_steps = configs.train_settings.max_train_steps
        if accelerator.is_main_process:
            logging.info(f"Using step-based training with max_train_steps={max_steps}")
    else:
        max_steps = configs.train_settings.num_epochs * steps_per_epoch
        if accelerator.is_main_process:
            logging.info(
                f"Using epoch-based training with num_epochs={configs.train_settings.num_epochs} (={max_steps} steps)"
            )

    # Determine checkpoint interval in steps
    if getattr(configs, "checkpoint_every_n_steps", None) is not None:
        checkpoint_interval_steps = configs.checkpoint_every_n_steps
        if accelerator.is_main_process:
            logging.info(f"Checkpointing every {checkpoint_interval_steps} steps")
    else:
        checkpoint_interval_steps = configs.checkpoints_every * steps_per_epoch
        if accelerator.is_main_process:
            logging.info(
                f"Checkpointing every {configs.checkpoints_every} epochs (={checkpoint_interval_steps} steps)"
            )

    # Determine validation interval in steps
    if getattr(configs.valid_settings, "validate_every_n_steps", None) is not None:
        validation_interval_steps = configs.valid_settings.validate_every_n_steps
        if accelerator.is_main_process:
            logging.info(f"Validating every {validation_interval_steps} steps")
    else:
        validation_interval_steps = configs.valid_settings.do_every * steps_per_epoch
        if accelerator.is_main_process:
            logging.info(
                f"Validating every {configs.valid_settings.do_every} epochs (={validation_interval_steps} steps)"
            )

    # Maybe monitor resource usage during training.
    prof = None
    profile_train_loop = configs.train_settings.profile_train_loop

    if profile_train_loop:
        from pathlib import Path

        train_profile_path = os.path.join(result_path, "train", "profile")
        Path(train_profile_path).mkdir(parents=True, exist_ok=True)
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=30, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(train_profile_path),
            profile_memory=True,
        )
        prof.start()

    # Use this to keep track of the global step across all processes.
    # This is useful for continuing training from a checkpoint.
    global_step = start_global_step

    # Initialize adaptive loss coefficients (persistent across epochs)
    adaptive_loss_coeffs = {
        "mse": 1.0,
        "backbone_distance": 1.0,
        "backbone_direction": 1.0,
        "binned_direction_classification": 1.0,
        "binned_distance_classification": 1.0,
        "vq": 1.0,
        "ntp": 1.0,
        "tik_tok_padding": 1.0,
    }

    best_valid_metrics = {
        "gdtts": 0.0,
        "mae": 1000.0,
        "rmsd": 1000.0,
        "lddt": 0.0,
        "loss": 1000.0,
        "tm_score": 0.0,
        "perplexity": 1000.0,
        "padding_accuracy": 0.0,
    }

    # Main training loop - now step-based instead of epoch-based
    current_epoch = start_epoch
    last_checkpoint_step = global_step
    last_validation_step = global_step

    while global_step < max_steps:
        start_time = time.time()
        training_loop_reports = train_loop(
            net,
            train_dataloader,
            current_epoch,
            adaptive_loss_coeffs,
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            configs=configs,
            logging=logging,
            global_step=global_step,
            max_steps=max_steps,
            steps_per_epoch=steps_per_epoch,
            writer=train_writer,
            result_path=result_path,
            profiler=prof,
            profile_train_loop=profile_train_loop,
        )

        if profile_train_loop:
            prof.stop()
            logging.info("Profiler stopped, exiting train epoch loop.")
            break

        end_time = time.time()
        training_time = end_time - start_time
        fractional_epoch = training_loop_reports["fractional_epoch"]

        # Only log training results to console if configured to do so
        if getattr(configs, "console_log_training", True):
            logging.info(
                f'epoch {fractional_epoch:.2f} ({training_loop_reports["counter"]} steps) - time {np.round(training_time, 2)}s, '
                f'global steps {training_loop_reports["global_step"]}/{max_steps}, loss {training_loop_reports["loss"]:.4f}, '
                f'rec loss {training_loop_reports["rec_loss"]:.4f}, '
                f'mae {training_loop_reports["mae"]:.4f}, '
                f'rmsd {training_loop_reports["rmsd"]:.4f}, '
                f'gdtts {training_loop_reports["gdtts"]:.4f}, '
                f'tm_score {training_loop_reports["tm_score"]:.4f}, '
                f'ntp loss {training_loop_reports["ntp_loss"]:.4f}, '
                f'perplexity {training_loop_reports.get("perplexity", float("nan")):.2f}, '
                f'padding acc {training_loop_reports.get("padding_accuracy", float("nan")):.4f}, '
                f'vq loss {training_loop_reports["vq_loss"]:.4f}, '
                f'activation {training_loop_reports["activation"]:.1f}'
            )

        global_step = training_loop_reports["global_step"]
        # Update adaptive coefficients from training loop
        adaptive_loss_coeffs = training_loop_reports.get(
            "adaptive_loss_coeffs", adaptive_loss_coeffs
        )
        accelerator.wait_for_everyone()

        # Step-based checkpointing
        if global_step - last_checkpoint_step >= checkpoint_interval_steps:
            tools = dict()
            tools["net"] = net
            tools["optimizer"] = optimizer
            tools["scheduler"] = scheduler

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # Set the path to save the models checkpoint.
                model_path = os.path.join(checkpoint_path, f"step_{global_step}.pth")
                save_checkpoint(
                    int(fractional_epoch),
                    model_path,
                    accelerator,
                    net=net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    configs=configs,
                    global_step=global_step,
                )
                logging.info(
                    f"\tcheckpoint models at step {global_step} (epoch {fractional_epoch:.2f}) in {model_path}"
                )

            last_checkpoint_step = global_step

        # Step-based validation
        if global_step - last_validation_step >= validation_interval_steps:
            start_time = time.time()
            valid_loop_reports = valid_loop(
                net,
                valid_dataloader,
                fractional_epoch,
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                configs=configs,
                logging=logging,
                global_step=global_step,
                writer=valid_writer,
                result_path=result_path,
            )
            end_time = time.time()
            valid_time = end_time - start_time
            accelerator.wait_for_everyone()

            # Only log validation results to console if configured to do so
            if getattr(configs, "console_log_validation", True):
                logging.info(
                    f'validation epoch {fractional_epoch:.2f} ({valid_loop_reports["counter"]} steps) - time {np.round(valid_time, 2)}s, '
                    f'step {global_step}/{max_steps}, '
                    f'loss {valid_loop_reports["loss"]:.4f}, '
                    f'rec loss {valid_loop_reports["rec_loss"]:.4f}, '
                    f'mae {valid_loop_reports["mae"]:.4f}, '
                    f'rmsd {valid_loop_reports["rmsd"]:.4f}, '
                    f'gdtts {valid_loop_reports["gdtts"]:.4f}, '
                    f'tm_score {valid_loop_reports["tm_score"]:.4f}, '
                    f'ntp loss {valid_loop_reports["ntp_loss"]:.4f}, '
                    f'perplexity {valid_loop_reports.get("perplexity", float("nan")):.2f}, '
                    f'padding acc {valid_loop_reports.get("padding_accuracy", float("nan")):.4f}, '
                    f'vq loss {valid_loop_reports["vq_loss"]:.4f}, '
                    f'activation {valid_loop_reports["activation"]:.1f}'
                    # f'lddt {valid_loop_reports["lddt"]:.4f}'
                )

            # Check valid metric to save the best model
            if valid_loop_reports["rmsd"] < best_valid_metrics["rmsd"]:
                best_valid_metrics["gdtts"] = valid_loop_reports["gdtts"]
                best_valid_metrics["mae"] = valid_loop_reports["mae"]
                best_valid_metrics["rmsd"] = valid_loop_reports["rmsd"]
                best_valid_metrics["loss"] = valid_loop_reports["loss"]
                best_valid_metrics["tm_score"] = valid_loop_reports["tm_score"]
                best_valid_metrics["perplexity"] = valid_loop_reports.get(
                    "perplexity", float("nan")
                )
                best_valid_metrics["padding_accuracy"] = valid_loop_reports.get(
                    "padding_accuracy", float("nan")
                )

                tools = dict()
                tools["net"] = net
                tools["optimizer"] = optimizer
                tools["scheduler"] = scheduler

                accelerator.wait_for_everyone()

                # Set the path to save the model checkpoint.
                model_path = os.path.join(checkpoint_path, "best_valid.pth")
                save_checkpoint(
                    int(fractional_epoch),
                    model_path,
                    accelerator,
                    net=net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    configs=configs,
                    global_step=global_step,
                )
                logging.info(
                    f"\tsaving the best models at step {global_step} (epoch {fractional_epoch:.2f}) in {model_path}"
                )
                logging.info(f'\tbest valid rmsd: {best_valid_metrics["rmsd"]:.4f}')

            last_validation_step = global_step

        # Increment current_epoch for next iteration
        current_epoch += 1

        # Check if we should stop early
        if training_loop_reports.get("early_stop", False):
            logging.info(f"Early stopping triggered at step {global_step}")
            break

    logging.info("Training is completed!\n")

    # log best valid gdtts
    logging.info(f"best valid gdtts: {best_valid_metrics['gdtts']:.4f}")
    logging.info(f"best valid tm_score: {best_valid_metrics['tm_score']:.4f}")
    logging.info(f"best valid rmsd: {best_valid_metrics['rmsd']:.4f}")
    logging.info(f"best valid mae: {best_valid_metrics['mae']:.4f}")
    logging.info(f"best valid perplexity: {best_valid_metrics['perplexity']:.2f}")
    logging.info(
        f"best valid padding accuracy: {best_valid_metrics['padding_accuracy']:.4f}"
    )
    logging.info(f"best valid loss: {best_valid_metrics['loss']:.4f}")

    if accelerator.is_main_process:
        train_writer.close()
        valid_writer.close()

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()
    torch.cuda.empty_cache()
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQ-VAE models.")
    parser.add_argument(
        "--config_path",
        "-c",
        help="The location of config file",
        default="./configs/config_vqvae.yaml",
    )
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
