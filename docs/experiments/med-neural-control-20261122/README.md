# Medvedev–Struff native neural control

For the locked Med000 mixed-mode passage, encode the original 240 ticks from zero at native 50 fps. Freeze DCVC-UF at Q24, 480×272, before inspecting quality. Inspect price only against the anticipated 21.5–23 kbps warm stream. If budget allows an adjacent price below/above, add only one QP selected from measured rates and past native curves. Evaluate the strongest affordable stream at the actual jointly delivered price. A rate that is too distant remains a separate rate point. Always retain the source geometry, four byte-identical independent receiver passes, all underlying coded bytes, and all 240 ticks. Primary readout uses the common source-defined scoring protocol when the joint receiver completes. GPU budget at most 2 × 2,150 s, 8 GiB shared Ada, each job ends by 15:00 UTC.

Fleet Q24 request `20261122T093041Z-95b7972b` uses the fresh unchanged native/legacy qualified decoder, revision 239570b4, with 48-tick smoke and 240-tick full stage.
