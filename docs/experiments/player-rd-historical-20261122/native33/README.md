# Ordinary 33 kbps control for the eight-window player comparison

Freeze one additional ordinary stream per missing source/scale family before scoring. Reuse Alcaraz–Hurkacz 034's actual CRF61, 38.946 kbps half-resolution stream from job `20261121T235319Z-0a6b4348`. Its 32.795 kbps CRF62 stream is the existing lower neighbor. For 50-fps Medvedev–Struff 019 and 024, and Nadal–Kyrgios 000 and 001, add CRF65 at quarter resolution (480×272); for Nadal–Kyrgios 002 use CRF67 at the same size because that window still used 37.92 kbps at CRF64. These settings come from the original rates before any new quality score. The source is the unmodified broadcast.

Each CPU worker reproduces the original 240-frame YUV conversion and the historical CRF63 IVF **and decoded YUV** digests, then publishes the new native IVF. Smoke uses 24 native frames on one CPU and both real encodes. Require a conservative forecast under 600 seconds; total cap 900 CPU-seconds per job. Start with Medvedev–Struff 019 before dispatching the remaining four. Each stream must be independently scored on the original 80 W7 source ticks and regional supports. Only use interpolation with measured points above and below the target on the same spatial-resolution curve. Keep a miss in the coverage table.

Medvedev–Struff 019 pilot `20261122T013055Z-0bc0e829` was submitted against the registered settings and original source. Its separate full-stage and total caps are 600 and 900 CPU-seconds.

The first 019 pilot stopped on input-module import before encoding. The scoped snapshot omitted the existing `tools.datasets` dependency. The corrected specification keeps all metrics, codecs and frame identities fixed and reserves 20 CPU-seconds for that lifecycle. Include `tools` in each subsequent committed source snapshot.

Corrected Medvedev–Struff 019 pilot `20261122T013254Z-990e05a0` submitted; snapshot includes native worker, shared sources and prior dataset tools.

The 019 CPU pilot passed full validation: exact historical source conversion, historical CRF63 stream and independently decoded YUV, 240 verified frames. The new CRF65 stream costs **29.705 kbps**, below the original 34.730 kbps CRF63 anchor. The full two-encode stage took 5.30 seconds on its allocated CPU. Archive SHA256 `2e772912b7ab0192b208cae4309f9c7c8c5694745e99d71a9ba15bff70c73b28`. This admits the four other registered streams. [Producer receipt](med019-result.json)

Registered window med024 campus CPU job 20261122T013630Z-b4fdca30 with the original common-control protocol.

Registered window nad000 campus CPU job 20261122T013655Z-fb4f785b with the original common-control protocol.

Registered window nad001 campus CPU job 20261122T013721Z-84ecaad4 with the original common-control protocol.

Registered window nad002 campus CPU job 20261122T013746Z-88b55622 with the original common-control protocol.

All remaining ordinary encodes passed the original bitstream, decoded-frame and source checks: med024 at 29.780 kbps, nad000 at 30.747 kbps, nad001 at 30.567 kbps, nad002 at 30.243 kbps. All four physically bracket 33 kbps with the existing higher ordinary encodes. Their W7 quality scores come from the separate regional judges. [Producer archives and checksums](additional-archives.json)
