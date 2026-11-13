# Hotel SSH diagnosis — November 10 local / November 11 UTC

The user also cannot log in from their terminal while staying in a hotel in
Brazil. Read-only diagnosis ran around 00:38–00:45 UTC on November 11, 2026.
No VPN, SSH configuration, credential, server, manuscript or research-job
setting was changed. No research job was inspected, launched or replayed.

## Observations

- macOS routes the default IPv4 connection through `utun4` (10.5.0.2);
  NordVPN and its helper are running. Hotel Wi-Fi is `en0` (10.125.161.100).
  The effective SSH configuration uses public GPU hostnames, port 22, account
  `emanuele`, and no ProxyJump or ProxyCommand.
- DNS resolves the GPU hosts to public university addresses. TCP connections
  to gpu5 and gpu3 returned OpenSSH banners in approximately 0.51 seconds.
  GitHub SSH and university HTTPS were also reachable through the default path.
- gpu5 completed SSH key exchange, matched its previously saved host key,
  received authentication methods, offered the existing Ed25519 public key,
  and then stopped receiving a response. The diagnostic connection was
  explicitly terminated locally; no remote command began.
- gpu1, gpu3 and gpu2 likewise reached `Offering public key` / `wait for reply`
  without receiving key acceptance or rejection. Bounded diagnostics then
  disconnected. There was no successful authentication or remote command.
- On **both gpu2 and gpu3**, disabling public-key authentication for one
  diagnostic connection returned the expected `Permission denied
  (publickey,keyboard-interactive)` promptly (4.60 and 1.75 seconds). These
  deliberate batch-mode negative controls do not test a password or establish
  that keyboard-interactive login would succeed.
- A separate gpu5 connection received **`Exceeded MaxStartups`** during banner
  exchange. This is a server response, not a local SSH configuration error.
- A single gpu3 connection bound to hotel Wi-Fi with macOS `nc -b en0` timed
  out during banner exchange after seven seconds. HTTPS to www.aau.at bound
  to the same interface timed out after five seconds. The default VPN route
  worked for these connection stages. Interface binding does not establish
  why direct traffic failed: VPN enforcement, hotel filtering, routing, or a
  different network issue remain possible. NordVPN was not disconnected.

## Assessment and next action

The default VPN path permits DNS, TCP, SSH encryption and initial authentication
messages. A blanket hotel port-22 block does not explain the observed stall on
that path. The repeatable public-key-specific stall on multiple servers points
to server-side key lookup or a shared authentication dependency. A stalled
NFS home / `.ssh/authorized_keys` lookup is a leading hypothesis because the
fleet shares home storage; **it has not been confirmed from server logs**.
Source-dependent policy or a VPN-path problem is not conclusively excluded.

OpenSSH documents that `MaxStartups` limits concurrent unauthenticated
connections, and that the default authorized-key files are in the user's home.
See the [official sshd_config manual](https://man.openbsd.org/sshd_config#MaxStartups)
and [AuthorizedKeysFile documentation](https://man.openbsd.org/sshd_config#AuthorizedKeysFile).
The default file location is not proof of these servers' actual configuration.

Stop additional connection retries. An ITEC administrator with an existing
session or console should inspect SSH authentication logs, unauthenticated
sshd children and blocked-process wait channels, and shared-home/NFS health.
They should verify the actual AuthorizedKeysFile/AuthorizedKeysCommand and
account-directory dependencies. Do not blindly reboot GPU hosts, kill workers,
or increase MaxStartups; preserve other users' jobs and the two unresolved
PointStream jobs. No administrator has been contacted by this session.

A phone-hotspot test is an optional independent transport control if needed;
it should initially keep the VPN state unchanged. It cannot replace checking
the observed server authentication stall.

## Recovery and evidence

The fleet research block remains in force. After login and shared-home access
are restored, read each experiment entry and canonically inspect the exact
existing Ball and W8 jobs before any further admission. Their unknown states
and maximum reservations are unchanged. See
[the continuation](continuation-2026-11-10.md).

Raw bounded-client logs were recorded under
`/private/tmp/pointstream-hotel-ssh-debug-20261110/`; that temporary directory
was absent at the November11 handoff audit. A tracked compact
[receipt](experiments/fleet-hotel-ssh-diagnosis-20261110.json) records their
SHA256 hashes and diagnostic excerpts. Logs contain
public host/key identifiers, not private-key contents or passwords. No
diagnostic SSH connection or recurring polling is intentionally left running.
