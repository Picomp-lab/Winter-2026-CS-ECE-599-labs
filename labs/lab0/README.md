# Lab 0 — OSU COE HPC + VS Code Remote SSH Setup

This lab is a hands-on setup lab. You will:
- Get access to the OSU COE HPC cluster
- Connect via SSH (or MobaXterm on Windows)
- Set up scratch storage (`/nfs/hpc/share/<ONID>`)
- Use VS Code Remote-SSH on the cluster
- Request a GPU interactively and run `nvidia-smi`
- Connect VS Code directly to the allocated GPU node

## 0) Prerequisites
- Your OSU Network ID (ONID) / email prefix (used as your HPC username)
- A local installation of VS Code
- Internet access to connect to the cluster

---

## 1) Request and enable HPC access
1. Request COE HPC access (follow the COE IT instructions). https://it.engineering.oregonstate.edu/hpc 
2. Enable your HPC account in TEACH. https://teach.engr.oregonstate.edu/teach.php?type=want_auth 

> You must complete both steps before SSH login will work.

---

## 2) Connect to the COE HPC cluster (SSH)
From a terminal on your laptop:

```bash
ssh <ONID>@submit.hpc.engr.oregonstate.edu
```

> Replace `<ONID>` with your username (email prefix).

### Windows alternative: MobaXterm
If you use Windows, install **MobaXterm** and SSH to:
- **Host:** `submit.hpc.engr.oregonstate.edu`
- **Username:** `<ONID>`

---

## 3) Use scratch storage (IMPORTANT)

Your home directory has a limited quota. Use scratch for course work.

**Scratch path:**
```text
/nfs/hpc/share/<ONID>
```

## 3.1 Create a convenient symlink and work there

Run on the submit node:

```bash
ln -s /nfs/hpc/share/<ONID> hpc-share
cd hpc-share
```

> For this course, put your repo and work under `~/hpc-share/`.

---

## 3.2 Check disk usage (helpful for quota/debugging)

```bash
du -ah --max-depth=1 | sort -hr
```

## 4) VS Code Remote-SSH into the cluster

### 4.1 Install extensions (local laptop)

In **VS Code** (on your laptop), install:

- **Remote - SSH**
- **Python**

### 4.2 Connect VS Code to the submit node

In **VS Code**:

1. Open **Command Palette** → `Remote-SSH: Connect to Host…`
2. Enter:

```bash
ssh <ONID>@submit.hpc.engr.oregonstate.edu
```

### 4.3 Open your course folder (recommended)
Open:
    
```bash
~/hpc-share/Winter-2026-CS-ECE-599-labs
```

### 4.4 Open the integrated terminal in VS Code

Use:
- **Ctrl + J** (or **Terminal → New Terminal**)

## 5) Clone your course repo into scratch (recommended)
Run on the submit node:

```bash
cd ~/hpc-share
git clone https://github.com/Picomp-lab/Winter-2026-CS-ECE-599-labs.git
cd Winter-2026-CS-ECE-599-labs
```

## 6) (Optional) Clone the demo/tutorial repo from class

If your instructor provided a demo repo:
```bash
git clone https://github.com/Picomp-lab/Winter-2026-CS-ECE-599-labs.git
```

## 7) Request a GPU interactively (srun)

Request one GPU for 1 hour with 64 GB RAM:
```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgx2 --gres=gpu:1 --mem=64G --pty bash
```

Once you’re on the allocated node, run:
```bash
nvidia-smi
hostname -f
```
- `nvidia-smi` confirms the GPU is available.
- `hostname -f` prints the compute node hostname you were assigned.

## 8) Connect VS Code directly to the GPU node (recommended)

After you run hostname -f on the GPU node, copy the hostname and connect VS Code Remote-SSH to:
```bash
ssh <ONID>@<HOSTNAME_FROM_hostname_-f>
```
This attaches your editor + terminal directly to the GPU node you reserved.