# Sydney clean-base parity

The wider Sydney rate figure needs the donor-filled sender raster before AV1
encoding. Rebuild only that raster through the original source-to-donor path,
with the source, SAM features, poses, transmitted ball, OpenCV and FFmpeg
bound to the archived sender. The competing explanation for a mismatch is a
different native decode, source mask, donor registration or donor raster.

The first stage checks exact donor registration counts at native ticks 0, 120
and 239. The complete replay must match original 17-donor counts, repaired
area at each tick, holes, mask checksum and clean YUV SHA-256. It publishes
the verified 240-frame YUV420p raster at 960×544 with a receipt. Budget one CPU
at most 1,120 seconds across stages, no GPU. The submitted specification pins
the stage caps, inputs and deadline.

- [Original sender receipt](../test-joint-extension-20261122/native/match127-sender-result.json)
- [CPU replay specification](clean-parity-spec.json)
- [Registration 20261123T004555Z-ca0fa0a9](clean-parity-registration.json)
- [Verified full receipt](clean-parity-result.json)

The replay passed on gpu3. It matched all 17 registrations, repaired pixel
counts at all 240 ticks, hole counts, the mask checksum and the original clean
raster SHA-256 `810783b90f05fd6f7413a6ff2d9a1db4a8903c2d5bdce05a89ddbe81e07238e8`.
The validated artifact is 188,006,400 bytes. The smoke and full replay
entrypoints took 3.49 and 16.54 seconds respectively, excluding fleet staging.
