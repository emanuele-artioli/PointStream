# Medvedev–Struff native neural control

For the locked Med000 mixed-mode passage, encode the original 240 ticks from zero at native 50 fps. Freeze DCVC-UF at Q24, 480×272, before inspecting quality. Inspect price only against the anticipated 21.5–23 kbps warm stream. If budget allows an adjacent price below/above, add only one QP selected from measured rates and past native curves. Evaluate the strongest affordable stream at the actual jointly delivered price. A rate that is too distant remains a separate rate point. Always retain the source geometry, four byte-identical independent receiver passes, all underlying coded bytes, and all 240 ticks. Primary readout uses the common source-defined scoring protocol when the joint receiver completes. GPU budget at most 2 × 2,150 s, 8 GiB shared Ada, each job ends by 15:00 UTC.

Fleet Q24 request `20261122T093041Z-95b7972b` uses the fresh unchanged native/legacy qualified decoder, revision 239570b4, with 48-tick smoke and 240-tick full stage.

Q24 completed on gpu6, 12,704 bytes and 21.1733 kbps at 480×272, all four native receiver passes identical on all 240 frames. The originally forecast ordinary price is above this setting. Reassess a second price only against the achieved mixed-mode bill. [Native stream and decode identities](q24-result.json)

The source-denied carrier bill is now 13,940 changing bytes (23.2333 kbps warm), 9.73% above Q24. At fixed geometry on the historical Alcaraz–Hurkacz rate curve, Q24 to Q36 increased bitrate from 20.4117 to 39.771 kbps. Applying its log-rate slope to the new byte gap suggests +1.67 QP; choose the nearest integer Q26, before scoring this candidate. Keep the same 480×272 and all other settings. If it exceeds the warm ceiling, use Q24 as the affordable neural point and report Q26 separately. No additional rate search on this passage.

Q26 fleet job `20261122T095116Z-093dddbd`, source 05d97d059c46, 48-tick smoke then an independent native 240-tick receiver.
