#!/bin/bash
# Example commands to reproduce main H/C alignment results.
# Modify paths and arguments as needed for your environment.

# Full finetune from pretrained (strict protocol)
bash scripts/run_hc_alignment_strict_full_finetune.sh

# Full finetune from scratch baseline
bash scripts/run_hc_alignment_strict_scratch_full_finetune.sh

# Unfreezing ablation (last 3 blocks)
bash scripts/run_hc_alignment_strict_unfreeze_last3.sh

# Few-shot experiments
bash scripts/run_hc_alignment_strict_fewshot.sh
