# Court cleanup diagnostic

On the opened 240-frame Nadal–Kyrgios development rally, test whether two other reviewed frames provide unoccluded court pixels beneath a player at an independently labeled frame. All three ticks 10, 90 and 170 were reviewed without proposals. At 960×544, ORB court-feature matches and a RANSAC homography align each donor to the target, excluding the union of its two reviewed player masks. The masks are resized from 1920×1080 and dilated with a 17×17 elliptical kernel. Replace each pixel in the target mask by the first aligned donor with an unmasked pixel. Keep target pixels for which neither donor is available. Run target 90 with donors 10 then 170, and target 10 with donors 90 then 170. The inputs are the previously qualified CRF 48 decoded base and the human-reviewed mask export, identified by SHA-256 in the receipt.

At tick 90 the target contains 11,066 masked pixels and 719 remain uncovered after both donors. At tick 10, 706 of 10,423 remain uncovered. The tick-90 output visibly retains a player fragment and racket. The reviewed masks cover player bodies and clothing, and do not establish racket, ball, shadow or halo removal. This is an exploratory two-frame check, not a clean-background codec result. It uses decoded future frames and human labels as free donor information. A receiver would need a background representation, timing, updates and any correction data to be delivered and charged, and full reconstruction quality assessed across the rally. Homography does not guarantee accurate alignment in the presence of changing court appearance or camera motion.

Reproduce with NumPy 1.26.4 and OpenCV 4.10.0, providing the private decoded YUV and reviewed JSON at their recorded local paths. Output is written to a new directory, leaving inputs untouched.

```sh
python docs/experiments/court-cleanup-diagnostic-20261118/diagnostic.py \
  --base /Users/manu/Datasets/pointstream-submission-20261118/ps-native-ctl-20261118-full/plain-48/decoded.yuv \
  --labels /Users/manu/Datasets/pointstream-submission-20261118/foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json \
  --output /private/tmp/ps-court-cleanup-new
```

The [receipt](receipt.json) records input identities, OpenCV version, target and donor ticks, homographies, inlier counts, filled pixels and residuals. Source video SHA-256 is `b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d`. Display-only before and after images are local artifacts, not a published reference or a bitrate measurement.
