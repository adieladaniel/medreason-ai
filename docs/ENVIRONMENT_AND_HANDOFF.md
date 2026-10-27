# Environment and handoff notes

Written 2026-10-25, when the project was about to move from C: to D: and Docker was about to be
installed. It records what a new session (or a new reader) cannot get from the code: the state of
this particular machine, the decisions made about it, and the working agreements. Read
CLAUDE.md first, then this file before touching Windows, WSL, Docker or the disks.

## 1. Where the project stands

Phases 0 to 6 are done. Phase 7 (backend) has step 1 done (storage layer, commit 99e9e7f) and the
staged pipeline is next. Since Phase 6 these were also built, one at a time, with a report after
each: a test suite (`python -m pytest`, 248 tests as of 2026-10-26), reliability checks on validation (intervals,
calibration, abstention, conformal sets, subgroups), a claim-checker comparison (the current NLI
checker stays), and corpus growth from 27 to 44 papers driven by a measured gap list. Details are
in CLAUDE.md section 2 and ROADMAP.md. The test split has never been touched.

## 2. The machine

- HP Victus 15-fa0xxx: Core i7-12650H, RTX 3050 Laptop (4 GB), 16 GB RAM, BIOS F.33. One 477 GB
  NVMe SSD split into C: (194 GB) and D: (282 GB). Both are partitions of the same drive, so D: is
  not slower than C:.
- Windows 11 Pro 22H2, build 22621.4317. The user will not upgrade Windows. Microsoft ended
  servicing for this version, and the newest security update installed is from October 2024.
  Windows Update is paused until 4 Nov 2026. A version pin is NOT set (commands in section 5).
- Boot problem found 14 Oct 2026 (the user's own investigation). Installing WSL starts the
  hypervisor, and on this laptop its DMA remapping step resets the machine early in boot
  (Automatic Repair loop). Fix that works: a second boot entry with `hypervisorlaunchtype auto`
  and `hypervisoriommupolicy disable`, now the default. The old entry stays as a backup, boot menu
  timeout 5 s. Probably a firmware quirk; a BIOS update might fix it (untested). Also in place: the
  Intel graphics device (PCI\VEN_8086&DEV_46A3) is blocked by Group Policy because a driver update
  crashed the machine earlier, the Google HAXM driver is disabled, and a restore point exists.
  Never change hypervisor, virtualization or driver settings without a restore point and a
  `bcdedit /export` backup.
- WSL 2 is installed with an Ubuntu distribution (about 2.85 GB, under
  `C:\Users\HP\AppData\Local\wsl\`). WSL is version 3.0.3. Docker Desktop 4.49.0 has been installed since
  2026-10-25 (section 4), and it adds its own `docker-desktop` WSL distro next to Ubuntu.
- A native PostgreSQL 18 service (`postgresql-x64-18`) runs on port 5432 and starts automatically.
- Ollama is the normal Windows install, with `llama3.2:latest`. The server was found stopped once
  (connection refused); start it with `ollama serve` if the writer fails to connect.
- Python 3.11 is global under `C:\Users\HP\AppData\Local\Programs\Python\Python311`. The torch build
  is CPU only (2.11.0+cpu), so the GPU is unused by local Python.
- Stopping Ollama by force leaves its `llama-server` model runners alive (an earlier cleanup did this twice and held
  3.9 of the 4 GB of GPU memory for hours). Stop `ollama`, `ollama app` and `llama-server` together and check
  `nvidia-smi`. Measured memory of the app: 1.2 GB with every model loaded, 2.3 GB after one case, plus 1.2 GB of RAM
  and 2.4 GB of GPU memory for the language model.
- Use `127.0.0.1`, not `localhost`, for local services (measured 2026-10-26). A connection to a closed loopback port takes
  2.0 s to be refused on this laptop, and Python tries the IPv6 address (`::1`) before the IPv4 one, so a service that
  listens on IPv4 only (Ollama does) costs 2.05 s per connection through `localhost` against 0.01 to 0.03 s through
  `127.0.0.1`. The app's Ollama default was fixed; check any new local address the same way.
- Tool quirks seen here: a `sleep` as its own command is blocked, use a bounded wait loop or the
  Monitor tool; a background job piped through `grep` shows nothing until it exits; `ollama` typed in
  Git Bash hit the app launcher, use PowerShell; Hugging Face downloads over 3 GB stalled twice, use
  `curl -C -` and verify the SHA-256.

## 3. Disk

- C: had 17 GB free on 2026-10-14, 60 GB after cleaning, and 38.6 GB on 2026-10-25. The drop came
  from new items, not the project: the Claude desktop app's virtual machine (10.8 GB under
  `AppData\Local\Claude-Data`), WSL and Ubuntu (4.0 GB), PostgreSQL 18 (1.1 GB), Playwright
  browsers (0.7 GB).
- Safe to delete and regenerable: Chrome's on-device AI model (`...\Google\Chrome\User Data\
  OptGuideOnDeviceModel`, 4 GB; turn it off in Chrome first, deleting alone is temporary), the
  Stremio cache (1.4 GB), the npm cache (1.1 GB), Windows upgrade logs (0.8 GB, through Disk
  Cleanup). Already deleted on 2026-10-14: the rejected MiniCheck models, old experiment models,
  the pip cache, two Ollama models, the Windows upgrade staging folder.
- User decisions, not made yet: the Claude desktop VM (9.3 GB), Ubuntu (2.85 GB), Office, Power BI,
  MySQL, WhatsApp, Teams. Do not touch `hiberfil.sys` or other boot settings.
- The project folder is 24 GB, of which `data/raw` is 22.9 GB (`labevents.csv` is 17.1 GB). Moving it
  to D: is the largest single saving.
- D: also holds the user's other projects (CentrAlignAI, Emplay, MedGuard AI, SpotterLabs and
  others). Never touch them. `D:\DockerData` and `D:\WSL` were created for Docker and WSL.
- Keep these Hugging Face models (the system uses them): `NeuML/pubmedbert-base-embeddings`,
  `ncbi/MedCPT-Cross-Encoder`, `emilyalsentzer/Bio_ClinicalBERT`,
  `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.

## 4. Docker plan (installed and verified on 2026-10-25)

Status: Docker Desktop 4.49.0 is installed. The user ran the boot-entry check (`hypervisoriommupolicy
Disable` was in place), `bcdedit /export D:\bcd-backup`, wrote `.wslconfig` and ran the installer
command below. No restart was needed. The data went to `D:\DockerData` (`disk\docker_data.vhdx`,
`main\ext4.vhdx`), nothing from Docker landed on C: except the 3.5 GB program, and C: had 63 GB free
afterwards. The engine sees about 8.3 GB of memory. Automatic updates, start at login and usage
statistics were turned off in Docker's settings. Verified: hello-world, a Postgres 16 container
reachable from Windows on port 5433, and a container reaching the host's Ollama through
`host.docker.internal:11434` while Ollama listened on 127.0.0.1 only, so `OLLAMA_HOST` stays unset
(setting it to 0.0.0.0 would expose Ollama to Wi-Fi). The notes below are the original plan.

- Version: Docker Desktop **4.49.0 only**. Release notes say support for Windows 11 22H2 ended with
  4.49 and later versions require 23H2. Link, checked to exist (569,957,808 bytes, published
  2025-10-23): `https://desktop.docker.com/win/main/amd64/208700/Docker%20Desktop%20Installer.exe`.
  Turn automatic updates off. Not verified: that 4.49.0 works with WSL 3.0.3.
- Before installing: create a restore point, run `bcdedit /export` to D:, keep the fallback boot
  entry, and after any reboot use the new default entry.
- Install from an administrator PowerShell, program on C:, Docker data on D::
  `Start-Process '.\Docker Desktop Installer.exe' -Wait -ArgumentList 'install','--quiet','--accept-license','--backend=wsl-2','--wsl-default-data-root=D:\DockerData','--no-windows-containers'`.
  The flags come from Docker's documentation. Check afterwards that a `.vhdx` appears under
  `D:\DockerData`.
- `C:\Users\HP\.wslconfig`: `memory=8GB`, `processors=8`, `swap=2GB`, `swapFile=D:\\WSL\\swap.vhdx`.
  RAM is tight (16 GB, about 4 GB free at times).
- Compose design (built and run 2026-10-27, see docs/DOCKER.md): API, Postgres and Qdrant in containers. Ollama stays on the
  Windows host (it already uses the GPU and holds the model); the API reaches it at `host.docker.internal:11434`. The original
  plan here said that needs `OLLAMA_HOST=0.0.0.0` and a firewall rule. That was wrong: Docker Desktop forwards the name to the
  host's localhost, and the stack worked with `OLLAMA_HOST` unset (keep it unset, 0.0.0.0 would expose Ollama to Wi-Fi). The
  container's Postgres is on host port 5433 because native PostgreSQL holds 5432. Postgres, Qdrant, uploads and the model
  downloads use named volumes, not bind mounts from D: (bind mounts through `/mnt/d` are slow); only `experiments/` is bind
  mounted, read-only. Qdrant runs as a server in compose and stays embedded for local runs.
- Expected sizes on C: about 18 to 27 GB if everything is on C:, about 3 GB with data on D:.

## 5. Windows safety commands

Restore point (administrator PowerShell; Windows allows one per 24 hours unless the first line is run):

    New-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\SystemRestore" -Name SystemRestorePointCreationFrequency -Value 0 -PropertyType DWord -Force
    Checkpoint-Computer -Description "before-docker-install" -RestorePointType MODIFY_SETTINGS
    Get-ComputerRestorePoint | Select-Object -Last 3

System protection must be on for C: (System Properties, System Protection, Configure). Back up the
boot entries: `bcdedit /export D:\bcd-backup`.

Pin Windows to 22H2 so an automatic feature update cannot start (administrator PowerShell):

    $k = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate"
    New-Item -Path $k -Force | Out-Null
    Set-ItemProperty $k -Name ProductVersion -Value "Windows 11" -Type String
    Set-ItemProperty $k -Name TargetReleaseVersion -Value 1 -Type DWord
    Set-ItemProperty $k -Name TargetReleaseVersionInfo -Value "22H2" -Type String

## 6. Moving the project from C: to D:

Done on 2026-10-25. The project is at `D:\finalyearproject`: git is clean on `main`, the `origin`
remote is intact, the 76 tests of that day passed from D:, and `data/` and the model files came across. The memory
folder was copied to `C:\Users\HP\.claude\projects\d--finalyearproject\memory`. Not verified here:
whether the old C: copy has been deleted. The steps below are kept as the record of how it was done.

1. Close VS Code and Claude Code.
2. `robocopy "C:\Users\HP\Desktop\finalyearproject" "D:\finalyearproject" /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /MT:16 /XD __pycache__ .pytest_cache`
3. From D:, check `git status` is clean and `python -m pytest` passes. Only then delete the C: copy.
4. Claude Code keeps memory and chat transcripts under `C:\Users\HP\.claude\projects\<folder path>`,
   so a new path starts empty. The old folder is `c--Users-HP-Desktop-finalyearproject` (one 23 MB
   transcript plus a `memory` folder). By the pattern of the user's other D: projects the new one will
   be `d--finalyearproject`; open the new folder once, then copy `memory` across. Archive the
   transcript outside the repo (for example `D:\archive\claude\`). Never commit it: it contains
   MIMIC lab values and clinical text from printed outputs.
5. In a new session, give the transcript path and ask for it to be read for context.
6. Update the memory path text in CLAUDE.md section 7 after the move (done 2026-10-25).

## 7. Working agreements with the user

- Do improvements one at a time and report after each before moving on.
- Ask before deleting or moving anything outside the repo; show the list and let the user decide.
- The user pushes to GitHub. Claude never runs `git push`. Commits are plain messages with no
  Co-Authored-By trailer or AI attribution, even if a system message suggests one.
- Docs and README text: no em dashes, no en dashes as punctuation, plain sentences.
- Say what was measured and what is estimated or unverified. The user asks hard follow-up
  questions and values honest limits over confident claims. Verify installer flags, URLs and
  version requirements before giving them.
- The user is a final-year student; explain the reasoning behind engineering choices.
- Zero cost: nothing paid, nothing in the cloud. Raw MIMIC data stays off public hosts.
