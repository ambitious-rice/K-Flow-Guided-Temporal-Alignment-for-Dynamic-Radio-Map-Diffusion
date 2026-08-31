# Artifact synchronization through Aliyun Drive

Read this reference when transferring datasets, models, checkpoints, logs, metrics, previews, or other experiment artifacts between the local node and a configured remote node. Do not use the cloud drive to deploy Git-tracked source code; source deployment remains `local source -> GitHub -> remote checkout`.

## Configuration and scope

Read `.agents/config.yaml` before a transfer. The verified RMDM setup uses `aliyunpan`, the `resource` drive, and the cloud root mapped for the selected server under `external_storage.server_roots`. `artifacts.local_results_root` and `artifacts.remote_results_root` provide the normal local and remote staging roots.

The configured default is an incremental, one-time sync. Use a distinct cloud subdirectory for every run or transfer. `aliyunpan sync` expects its paired directories to be exclusively owned by that sync task. Do not use the shared cloud root itself as a sync destination, and do not use `exclusive` policy unless deletion of extra destination files is explicitly intended and authorized.

## Login verification and user-guided recovery

Login state is local to each machine. Check it on the machine performing the transfer:

```bash
aliyunpan who
```

If no account is logged in, start the interactive flow:

```bash
aliyunpan login
```

The client prints an authorization URL that expires after five minutes and waits for Enter. Give that temporary URL to the user, ask them to finish browser authorization and the required scan, then send Enter only after they confirm. Re-run `aliyunpan who` to verify the login and selected drive. If the authorization expires or fails, cancel the waiting command and generate a fresh URL. Do not attempt to automate the scan, place credentials in commands or logs, or copy credential/configuration files between machines.

## One-time artifact transfer

Select only the artifacts needed for the task, stage them in a dedicated local directory, and use a run-specific cloud directory. The tested one-time upload pattern is:

```bash
aliyunpan sync start \
  -ldir "<staging-directory>" \
  -pdir "<cloud-root>/<run-id>" \
  -mode upload \
  -drive resource \
  -policy increment \
  -cycle onetime \
  -log true
```

To retrieve the same artifact set on the other machine, use a different empty local directory and change only the mode:

```bash
aliyunpan sync start \
  -ldir "<destination-directory>" \
  -pdir "<cloud-root>/<run-id>" \
  -mode download \
  -drive resource \
  -policy increment \
  -cycle onetime \
  -log true
```

After transfer, verify that the command exited successfully and inspect the expected files. Do not generate, compare, or require SHA/checksum manifests for artifact synchronization. Use an infinite sync cycle only when the user explicitly requests a persistent synchronization task; record that task and its paths in `.agents/runs/`.

Record significant artifact transfers in the associated run record: cloud path, local and remote paths, direction, command/options, and verification result. A short isolated connectivity test does not need a run record; delete only its exact, known test paths after verification.
