# Training on the desktop GPU

How to train `rl_replay.train_nn` on the RTX 3090 desktop, which runs Windows. Training runs inside WSL2 (Ubuntu on Windows), so the commands are the same as on server1, and the setup carries over if the desktop moves to Linux.

The training data (about 15 GB of `data/kaggle/frames/`) is built on server1 and copied over SSH. Results go back to server1 the same way.

## Part A: done by the owner in Windows

1. **Update the NVIDIA driver** with the NVIDIA app or from nvidia.com. Any driver from late 2024 or newer (version 560 or higher) works. WSL2 uses this Windows driver, so don't install an NVIDIA driver inside Ubuntu.
2. **Install WSL2 and Ubuntu.** Open PowerShell as administrator and run:
   ```
   wsl --install -d Ubuntu-24.04
   ```
   Restart when asked. Ubuntu opens and asks for a new Linux username and password. The password is for `sudo` inside Ubuntu.
3. **Check the GPU from Ubuntu.** In the Ubuntu window, run `nvidia-smi`. It should list the RTX 3090.
4. **Install Claude Code in Ubuntu** and sign in:
   ```
   curl -fsSL https://claude.ai/install.sh | bash
   claude
   ```

## Part B: done by Claude Code on the desktop

Run each step, check its result, and stop to report if one fails.

1. **Tools.** Make sure `git`, `rsync` and `curl` exist (`sudo apt install -y git rsync curl` if not; ask the owner before using sudo). Install uv:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Code.** `git clone https://github.com/Pinilla/rl-replay-ml ~/rl-replay-ml`
3. **SSH key for server1.** server1 (`192.168.4.24`, user `pinilla`) accepts SSH keys only. The owner's Windows key `bpini@BENPC` is already authorized. Copy it into Ubuntu, usually from `C:\Users\bpini\.ssh\`:
   ```
   mkdir -p ~/.ssh && cp /mnt/c/Users/bpini/.ssh/id_ed25519 /mnt/c/Users/bpini/.ssh/id_ed25519.pub ~/.ssh/
   chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519
   ssh -o StrictHostKeyChecking=accept-new pinilla@192.168.4.24 hostname
   ```
   The last command should print `server1`. If the key file has another name, list `/mnt/c/Users/*/.ssh/` to find it.
4. **Data.** Copy the training examples. Keep them in the Ubuntu file system (under `~`), not on `/mnt/c`, which is much slower from WSL2.
   ```
   mkdir -p ~/rl-replay-ml/data/kaggle/frames
   rsync -a --info=progress2 pinilla@192.168.4.24:rl-replay-ml/data/kaggle/frames/ ~/rl-replay-ml/data/kaggle/frames/
   ```
   About 15 GB: a few minutes on wired gigabit, longer on Wi-Fi.
5. **PyTorch with CUDA.**
   ```
   cd ~/rl-replay-ml && uv sync --extra gpu
   uv run --extra gpu python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   It should print `True` and `NVIDIA GeForce RTX 3090`.
6. **Quick test,** about a minute:
   ```
   uv run --extra gpu python -m rl_replay.train_nn --name mlp-gputest --shards 2 --max-rows 1000000 --epochs 2
   ```
   It should print `device: cuda (NVIDIA GeForce RTX 3090)`, two epoch lines and a results block.
7. **Full run.** Run it in the background with a log, since it can take a while:
   ```
   nohup uv run --extra gpu python -m rl_replay.train_nn --name mlp > train-mlp.log 2>&1 &
   ```
   Each epoch prints a line with `train_secs`. The run stops on its own when the validation score stops improving, at most 30 epochs.
8. **Send results to server1.**
   ```
   rsync -a models/mlp/ pinilla@192.168.4.24:rl-replay-ml/models/mlp/
   rsync -a train-mlp.log pinilla@192.168.4.24:rl-replay-ml/models/mlp/
   ```
   Then compare `models/mlp/results.json` with the LightGBM survival row in `RESULTS.md`. Don't push to GitHub from the desktop; server1 records results.

## Sequence model

`rl_replay.train_seq` is the same idea as the MLP, but it sees the last 4 s of play (8 moments, 0.5 s apart) instead of one moment. It uses the same data, so nothing new needs copying. Run it from `~/rl-replay-ml` after `git pull`:

1. **Quick test,** a minute or two:
   ```
   uv run --extra gpu python -m rl_replay.train_seq --name seq-gputest --shards 2 --max-rows 1000000 --epochs 2
   ```
2. **Full run:**
   ```
   nohup uv run --extra gpu python -m rl_replay.train_seq --name seq > train-seq.log 2>&1 &
   ```
   Each epoch prints `train_secs`. At the end it also scores `models/mlp` on the same test rows, under `mlp_same_rows` in `results.json`.
3. **Send results to server1:**
   ```
   rsync -a models/seq/ train-seq.log pinilla@192.168.4.24:rl-replay-ml/models/seq/
   ```

## If something goes wrong

- **`nvidia-smi` not found or no GPU in Ubuntu:** update the Windows NVIDIA driver, then run `wsl --shutdown` in PowerShell and reopen Ubuntu.
- **Out of memory while loading:** WSL2 gets half the PC's RAM by default (16 GB of 32). Loading needs about 10 GB. If that's too tight, create `C:\Users\bpini\.wslconfig` with these two lines, then run `wsl --shutdown`:
  ```
  [wsl2]
  memory=24GB
  ```
- **CUDA out of memory:** lower `--batch`, or train on part of the data with `--max-rows 15000000`.
- **Sequence model too slow** (over about 3 minutes per epoch): stop it and rerun with `--steps 6 --width 192`.
