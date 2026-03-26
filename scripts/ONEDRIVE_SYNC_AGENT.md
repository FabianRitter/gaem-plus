# OneDrive Sync Agent Instructions

## Purpose

This agent syncs GAEM+ research files between the NSCC cluster and OneDrive. The OneDrive folder mirrors the local project structure and serves as a cloud backup + collaboration point (Fabian edits paper readings on his laptop, syncs here).

## Paths

| Location | Path |
|----------|------|
| **Local** | `/data/projects/12004380/fabian/generalized_model_merging/` |
| **OneDrive** | `onedrive:Proposal-PhD/MERGING SPEECH ENCODERS RESEARCH/Unified Model Merging Last PhD Work 2026/` |
| **Sync script** | `scripts/onedrive_sync.sh` |

## What Syncs

| Item | Direction | Notes |
|------|-----------|-------|
| `GAEM_Research_Matrices.xlsx` | Push (local → OneDrive) | Updated after adding papers or changing statuses |
| `paper-readings/**/*.md` | Bidirectional | Fabian may edit on laptop (pull) or agent creates here (push) |
| `CLAUDE.md` | Push only | Agent reference, doesn't change on OneDrive |

**Never sync**: `gaem/`, `experiments/`, `scripts/`, `ssl-phase1/`, `results/`, `logfiles/`, `data/` — these are code/data that belong in git or are too large.

## Commands

```bash
# Push everything (Excel + readings + CLAUDE.md)
bash scripts/onedrive_sync.sh push

# Pull everything (Excel + readings from OneDrive → local)
bash scripts/onedrive_sync.sh pull

# Push only the Excel file (most common after adding papers)
bash scripts/onedrive_sync.sh push-excel

# Pull only paper-readings (when Fabian edited on laptop)
bash scripts/onedrive_sync.sh pull-readings

# Push only paper-readings (after agent creates new notes)
bash scripts/onedrive_sync.sh push-readings

# Check what differs
bash scripts/onedrive_sync.sh status
```

## Agent Workflow

### When user says "sync to OneDrive" or "push to OneDrive"
1. Run `bash scripts/onedrive_sync.sh push`
2. Confirm success

### When user says "pull readings" or "check OneDrive for new notes"
1. Run `bash scripts/onedrive_sync.sh pull-readings`
2. Check if any new .md files appeared in `paper-readings/`
3. If new files found, report what was pulled and offer to update the Excel

### When user says "update Excel and sync"
1. Make the requested Excel changes (via openpyxl)
2. Run `bash scripts/onedrive_sync.sh push-excel`
3. Also `git add -f GAEM_Research_Matrices.xlsx && git commit -m "Update research matrices" && git push`

### After adding papers to Excel
Always push to both OneDrive and GitHub:
```bash
bash scripts/onedrive_sync.sh push-excel
git add -f GAEM_Research_Matrices.xlsx && git commit -m "Update research matrices: [brief description]" && git push
```

### After creating a new paper reading note
1. Save the .md file in the correct `paper-readings/` subfolder
2. Push to OneDrive: `bash scripts/onedrive_sync.sh push-readings`
3. Commit to git: `git add paper-readings/ && git commit -m "Add reading note: [paper]" && git push`

## Conflict Resolution

If the same file was edited both locally and on OneDrive:
- **Excel**: Local version wins (it has the agent's updates). Push local → OneDrive.
- **Paper readings**: OneDrive version wins (Fabian's edits take priority). Pull OneDrive → local.
- Always ask the user before overwriting if unsure.
