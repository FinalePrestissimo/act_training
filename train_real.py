#!/usr/bin/env python3
import argparse
import os
import pickle
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from act_training.act_policy import ACTPolicy, CNNMLPPolicy
from act_training.data_utils import load_data, compute_dict_mean, detach_dict, set_seed


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    if policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    raise NotImplementedError(f"Unsupported policy_class: {policy_class}")


def make_optimizer(policy_class, policy):
    if policy_class in ["ACT", "CNNMLP"]:
        return policy.configure_optimizers()
    raise NotImplementedError(f"Unsupported policy_class: {policy_class}")


def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device, non_blocking=True)
    qpos_data = qpos_data.to(device, non_blocking=True)
    action_data = action_data.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad)


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    if not train_history or not validation_history:
        return
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label="train")
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label="validation")
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    device = config["device"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        print(f"\\nEpoch {epoch}")

        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for data in val_dataloader:
                forward_dict = forward_pass(data, policy, device)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")

        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")

        if (epoch + 1) % config["save_freq"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)

    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def parse_args():
    parser = argparse.ArgumentParser("Standalone ACT real-data trainer")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing episode_*.hdf5")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint output directory")
    parser.add_argument("--camera_names", type=str, default="cam_high,cam_right_wrist,cam_left_wrist")
    parser.add_argument("--num_episodes", type=int, default=-1, help="Use -1 to auto-count")
    parser.add_argument("--policy_class", type=str, default="ACT", choices=["ACT", "CNNMLP"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_freq", type=int, default=2000)

    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--state_dim", type=int, default=14)

    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--pretrained_backbone", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = os.path.realpath(args.dataset_dir)
    ckpt_dir = os.path.realpath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    camera_names = [x.strip() for x in args.camera_names.split(",") if x.strip()]
    if len(camera_names) == 0:
        raise ValueError("camera_names must contain at least one camera")

    if args.num_episodes <= 0:
        num_episodes = len(list(Path(dataset_dir).glob("episode_*.hdf5")))
    else:
        num_episodes = args.num_episodes

    if num_episodes <= 1:
        raise ValueError(f"Need at least 2 episodes for train/val split, got {num_episodes}")

    for i in range(num_episodes):
        p = os.path.join(dataset_dir, f"episode_{i}.hdf5")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.policy_class == "ACT":
        policy_config = {
            "lr": args.lr,
            "num_queries": args.chunk_size,
            "chunk_size": args.chunk_size,
            "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": args.lr_backbone,
            "backbone": args.backbone,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "nheads": args.nheads,
            "camera_names": camera_names,
            "weight_decay": args.weight_decay,
            "state_dim": args.state_dim,
            "pretrained_backbone": args.pretrained_backbone,
        }
    else:
        policy_config = {
            "lr": args.lr,
            "lr_backbone": args.lr_backbone,
            "backbone": args.backbone,
            "num_queries": 1,
            "camera_names": camera_names,
            "weight_decay": args.weight_decay,
            "state_dim": args.state_dim,
            "pretrained_backbone": args.pretrained_backbone,
        }

    config = {
        "num_epochs": args.num_epochs,
        "ckpt_dir": ckpt_dir,
        "state_dim": args.state_dim,
        "policy_class": args.policy_class,
        "policy_config": policy_config,
        "seed": args.seed,
        "save_freq": args.save_freq,
        "device": device,
    }

    train_loader, val_loader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        args.batch_size,
        args.batch_size,
    )

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    best_epoch, min_val_loss, best_state_dict = train_bc(train_loader, val_loader, config)

    best_ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, best_ckpt_path)
    print(f"Best ckpt saved to {best_ckpt_path}, val loss {min_val_loss:.6f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
