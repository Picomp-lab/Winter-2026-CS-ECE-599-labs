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
1. Request COE HPC access (follow the COE IT instructions). :contentReference[oaicite:2]{index=2}  
2. Enable your HPC account in TEACH. :contentReference[oaicite:3]{index=3}  

> You must complete both steps before SSH login will work.

---

## 2) Connect to the COE HPC cluster (SSH)
From a terminal on your laptop:

```bash
ssh <ONID>@submit.hpc.engr.oregonstate.edu

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

## 3.1 Create a convenient symlink and work there

Run on the submit node:

```bash
ln -s /nfs/hpc/share/<ONID> hpc-share
cd hpc-share
> For this course, put your repo and work under `~/hpc-share/`.

---

## 3.2 Check disk usage (helpful for quota/debugging)

```bash
du -ah --max-depth=1 | sort -hr

