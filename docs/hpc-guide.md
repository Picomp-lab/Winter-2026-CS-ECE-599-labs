# OSU COE HPC + VS Code (Remote SSH) Guide

This guide walks you through:
1) Getting HPC access
2) Connecting to the cluster via SSH and VS Code
3) Using scratch storage
4) Requesting a GPU job and attaching VS Code to the compute node

---

## 1) Request HPC access
- Request COE HPC access from the COE IT page.
- Enable your HPC account in TEACH.

(Links provided in the course announcement / Lab 0 handout.)

---

## 2) Connect to the COE HPC cluster
From a terminal:

```bash
ssh <ONID>@submit.hpc.engr.oregonstate.edu
