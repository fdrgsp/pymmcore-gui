# Pixel calibration: reference-patch evaluation

## Decision

Keep the current **75% reference patch** and **12% full-image fitting travel**.
The latter is `crop_fraction * target_shift_fraction = 0.75 * 0.16` per axis,
before any safety-radius reduction. Keep algorithm version **1**.

This combination retained more usable samples in the synthetic comparison
than 25% or 50% patches at the same travel distance. The larger patch was
particularly useful for blurred, sparse, and peripheral features. It is not
universally best: the 25% and 50% patches were faster and handled strong
uneven illumination better. No production defaults or acceptance thresholds were
changed for this evaluation.

[Different from Java: retain Python's larger fractional patch and shorter
fitting moves. The shared strategy remains reference-patch tracking and
Fourier cross-correlation followed by an affine fit.]

## Scope and reproducibility

Evaluated on 2026-09-16 using pymmcore-gui commit `6057672`, with the local
algorithm-version metadata set to `1`. Java source reference: Micro-Manager
commit `68a60c16c`, `AutomaticCalibrationThread.java`.

The [benchmark](../../../../tests/benchmark_pixel_calibration_patches.py)
runs the actual Python calibration routine against a simulated camera/stage.
The Java calibrator was **not executed**. A 25% patch on the square,
power-of-two camera sizes used here has Java's patch dimensions, but still
uses Python's preprocessing, registration, fitting, and acceptance checks.
These results isolate patch-size choices within Python; they do not measure
Java's success rate or speed.

[Partly same as Java: the 25% variant matches its patch dimensions on these
images. Java also differs in windowing, peak refinement, probing, fitting,
and confidence checks, as described in the main calibration document.]

The main experiment comprised 648 runs:

- Camera: 256 by 256 pixels, cropped from a 512 by 512 sample field.
- Patches: 25%, 50%, and 75% of each image axis.
- Fitting travel: 8%, 12%, and 25% of each full image axis, independently
  varied from patch size. Holdout travel remained 65% of fitting travel.
- Six sample families and two negative-control families.
- Three sample seeds and three frame-noise seeds per configuration.
- Ground truth: 0.4 micrometres per pixel, 19-degree rotation, reflected Y.
- Default confidence thresholds, probing, repeated captures, drift
  correction, fitting, and holdouts; settling disabled in the simulation.
- Safety radius: 100 micrometres. It did not shorten these target moves.

A follow-up used 512 by 512 images, the same three patch fractions, 12%
travel, blurred/sparse/uneven-illumination samples, three sample seeds, and
two noise seeds: another 54 runs, for **702 calibration runs** in total.

The benchmark overrides only the target generator and capture clock inside
its execution context. Using fixed capture times avoids results depending on
CPU scheduling; it does not test physical timing or drift. Target-generator
parity was checked against the production formula, including the default
75%/12% configuration. Twenty-four cases were reproduced exactly with the
saved benchmark, excluding elapsed runtime.

Noise is deterministic by capture index and noise seed, pairing corresponding
frames across configurations. Different sample seeds change the positive
sample fields. The blank and periodic fields repeat across sample indices;
those controls are repeated checks, not independent sample realizations.

### Sample construction

Images are floating-point grayscale. Sample fields are translated before
cropping the camera view, avoiding artificial wrapping at camera borders.
Gaussian noise is added separately to each capture. Field contrast is
normalized using the central camera view.

| Family | Construction |
| --- | --- |
| Fine texture | Random field, Gaussian blur sigma 2 pixels |
| Blurred texture | Random field, Gaussian blur sigma 6 pixels |
| Noisy texture | Fine texture with noise sigma 0.5 instead of 0.15 |
| Sparse features | Random Gaussian spots; about 12 per camera field |
| Peripheral features | Fine texture with a blank central disk |
| Uneven illumination | Fine texture plus background under fixed shading |
| Periodic control | Two perpendicular 16-pixel-period cosines |
| Blank control | Noise without sample structure |

Spot widths range from 2 to 6 pixels. The central blank disk reaches 22% of
image width in radius, with a 5%-width transition to the surrounding texture.
The illumination field is a centered Gaussian with sigma 25% of image width,
a 0.2 baseline, and 0.8 amplitude; the sample has an added background of 3.
Default noise sigma is 0.15; the periodic control uses 0.02.

## Results at the current measurement distance

At 12% fitting travel on 256 by 256 images:

| Sample | 25% patch | 50% patch | 75% patch |
| --- | ---: | ---: | ---: |
| Fine texture | 9/9 | 9/9 | 9/9 |
| Blurred texture | 6/9 | 9/9 | 9/9 |
| Noisy texture | 9/9 | 9/9 | 9/9 |
| Sparse features | 0/9 | 3/9 | 6/9 |
| Peripheral features | 0/9 | 0/9 | 9/9 |
| Uneven illumination | 9/9 | 8/9 | 3/9 |
| Positive-sample total | **33/54** | **38/54** | **45/54** |

Entries count completed, accepted calibrations. The family mix is a stress
suite, not an estimate of success probabilities on real microscope samples.
Blank and periodic controls were rejected for every patch/travel combination.
The simulated stage returned to its origin in all 702 runs.

Across the accepted positive cases at 12% travel:

| Error measure | 25% patch | 50% patch | 75% patch |
| --- | ---: | ---: | ---: |
| Median affine-matrix error | 0.098% | 0.065% | 0.053% |
| Worst affine-matrix error | 0.253% | 0.271% | 0.383% |
| Median absolute pixel-size error | 0.045% | 0.050% | 0.030% |
| Worst absolute pixel-size error | 0.174% | 0.270% | 0.381% |

Affine error is the Frobenius norm of the matrix error divided by the norm
of the true matrix. Pixel-size error compares the fitted scalar with 0.4.
These are errors against known truth, not the routine's internal fit RMS.
The accepted subsets differ: accuracy figures must be read with the rejection
counts, not used alone to rank configurations.

For the 75% patch, the worst matrix error outside the uneven-illumination
family was 0.097%. Its larger worst-case error in the complete table comes
from uneven illumination, so a larger patch does not improve every metric.

### Repeatability

At the current 75%/12% setting, all three noise realizations of each positive
sample agreed on pass/fail. Across samples that passed all three runs, the
largest pixel-size range was **0.086% of the true pixel size**. The 512-pixel
follow-up also had consistent pass/fail outcomes within each sample.

The 50%/12% setting had one uneven-illumination sample pass twice and fail
once. The 25%/12% setting had consistent outcomes, but rejected more usable
samples overall. Three repetitions are a limited check, not a guarantee that
future captures cannot cross an acceptance threshold.

### Larger images

At 512 by 512 pixels and 12% travel:

| Sample | 25% patch | 50% patch | 75% patch |
| --- | ---: | ---: | ---: |
| Blurred texture | 6/6 | 6/6 | 6/6 |
| Sparse features | 0/6 | 2/6 | 4/6 |
| Uneven illumination | 6/6 | 6/6 | 4/6 |

The sparse-feature advantage and illumination disadvantage persisted. The
50% and 75% patches tied in total passes on this smaller follow-up suite.

## Why not increase the travel distance?

For the 75% patch on 256-pixel images:

| Full-image travel | Accepted positives | Worst affine error among passes |
| --- | ---: | ---: |
| 8% | 45/54 | 0.454% |
| 12%, current | 45/54 | 0.383% |
| 25% | 39/54 | 1.521% |

A centered 75% patch has only 12.5% of image width available for tracking
in either direction without clipping. The current 12% fitting move fits
inside that margin when the prediction is accurate. A 25% move exceeds it,
leaving appreciable displacement inside the extracted patches.

A separate noise-free registration check on a blurred sample illustrates the
bias. For a true displacement of `(64, 64)` pixels, the 75% patch returned
`(63.30, 63.25)`, approximately 1.03 pixels short in vector norm. The 25% and
50% patches returned `(64, 64)`. At the current travel distance, all three
were within 0.043 pixels of truth in the same check.

Thus the clipped, windowed registration can be repeatable but biased. Some
25%-travel calibrations passed the existing checks despite a pixel-size error
as high as 1.489%. Keeping the current shorter travel is preferable to
adopting Java-like travel with the larger Python patch.

[Different from Java: Java's final corner offset is half the image dimension
minus its patch side. On a 256-pixel square image this is 64 pixels, or 25%.
Its smaller patch has sufficient tracking margin for those moves.]

## The illumination exception

With strong fixed illumination shading, the larger patch sometimes rejected
the initial reference pair despite zero actual motion. For two sample seeds,
the first-pair PSR was approximately 7.58 and 7.45, below the required 8;
measured shifts were zero and alignment errors were approximately 0.24 and
0.25. Smaller patches passed those reference checks.

The larger patch includes more of the smooth illumination variation, which
can dominate the correlation shape. Its confidence checks are therefore
sensitive to structure other than the features being tracked. This result
argues against claiming that 75% is always optimal or simply lowering the
confidence threshold. Background correction or deliberate patch selection
would need a separate evaluation, including ambiguous-sample rejection.

## Processing cost

Median time per cached-reference registration, with three warmups and ten
measured calls per patch size; order alternated between ascending and
descending patch size. Moving-image generation and reference initialization
were excluded. The predicted displacement was accurate and travel was 12%.

| Camera image | 25% patch | 50% patch | 75% patch |
| --- | ---: | ---: | ---: |
| 256 by 256 | 0.38 ms | 0.95 ms | 1.93 ms |
| 512 by 512 | 0.93 ms | 3.34 ms | 7.07 ms |
| 1024 by 1024 | 3.66 ms | 13.79 ms | 29.99 ms |

Environment: macOS 26.6.2, arm64, Python 3.13.3, NumPy 2.5.3. These are local
Python timings, not Java timings or end-to-end microscope calibration times.
Camera exposure, transfers, stage motion, and settling were not benchmarked.
At 1024 pixels, the 75% patch cost approximately 2.2 times the 50% patch and
8.2 times the 25% patch. This is a real cost, accepted here in exchange for
retaining more difficult samples.

## Follow-up: 60% patch

An additional 180 runs evaluated a 60% patch: 144 on the complete 256-pixel
suite and 36 on the 512-pixel follow-up. Both 9.6% travel, obtained by leaving
`target_shift_fraction = 0.16`, and 12% travel, equivalent to setting it to
0.20, were tested. All other settings and sample/noise seeds matched the
original experiment. Production defaults remain unchanged.

On 256 by 256 images:

| Sample | 60%, 9.6% travel | 60%, 12% travel | 75%, 12% travel |
| --- | ---: | ---: | ---: |
| Fine texture | 9/9 | 9/9 | 9/9 |
| Blurred texture | 9/9 | 9/9 | 9/9 |
| Noisy texture | 9/9 | 9/9 | 9/9 |
| Sparse features | 6/9 | 6/9 | 6/9 |
| Peripheral features | 3/9 | 2/9 | 9/9 |
| Uneven illumination | 6/9 | 6/9 | 3/9 |
| Positive-sample total | **42/54** | **41/54** | **45/54** |

All three peripheral-feature samples passed only one of three repetitions
with 60%/9.6%. Two passed one of three with 60%/12%; the third always failed.
All repetitions passed with 75%/12%. The smaller patch therefore increased
the pass/fail variability that this pipeline is intended to reduce.

Among samples with three successful repeats, the largest pixel-size range
was 0.489% of truth for 60%/9.6%, 0.446% for 60%/12%, and 0.086% for 75%/12%.
Worst accepted affine errors were respectively 0.822%, 0.467%, and 0.383%.
These maxima cover different accepted subsets, as in the original table.

The 512-pixel follow-up again shows sample dependence:

| Sample | 60%, 9.6% travel | 60%, 12% travel | 75%, 12% travel |
| --- | ---: | ---: | ---: |
| Blurred texture | 6/6 | 6/6 | 6/6 |
| Sparse features | 4/6 | 4/6 | 4/6 |
| Uneven illumination | 6/6 | 2/6 | 4/6 |
| Total | 16/18 | 12/18 | 14/18 |

All new negative-control runs were rejected, and the stage returned in all
180 new runs. The larger-image follow-up had no mixed pass/fail outcomes
within a sample for either 60% setting.

### Timing is not proportional to pixel count

A paired timing comparison used the same procedure as above, now selecting
only 60% and 75% patches. Travel for timing remained 12% for both:

| Camera image | 60% patch | 75% patch | Effect of using 60% |
| --- | ---: | ---: | --- |
| 256 by 256 | 1.41 ms | 1.93 ms | 27% faster |
| 512 by 512 | 8.84 ms | 7.24 ms | 22% slower |
| 1024 by 1024 | 38.62 ms | 30.23 ms | 28% slower |

The 60% patch contains approximately 36% fewer pixels, but its dimensions
are less favorable for FFT computation on these larger images. For 512 and
1024 pixels, its sides are 307 and 614, versus 384 and 768 at 75%. The former
contain the large prime factor 307; the latter contain only factors 2 and 3.
A separate forward-FFT timing check measured approximately 1.70 versus
0.90 ms for 307 versus 384, and 7.80 versus 4.00 ms for 614 versus 768.

Consequently, 60% is not a dependable speed optimization for this
implementation. Combined with the peripheral-feature repeatability results,
the evidence continues to support keeping 75% as the default. It does not
establish that 75% is best for every image size or sample.

Reproduce the follow-up with:

```sh
PYTEST_RUNNING=1 uv run python tests/benchmark_pixel_calibration_patches.py \
  --crops .6 --distances .096 .12 \
  --output /tmp/pixel_patch_60_results.jsonl

PYTEST_RUNNING=1 uv run python tests/benchmark_pixel_calibration_patches.py \
  --crops .6 --distances .096 .12 --size 512 --samples 3 --repeats 2 \
  --scenes blurred sparse illumination \
  --output /tmp/pixel_patch_60_512.jsonl

PYTEST_RUNNING=1 uv run python tests/benchmark_pixel_calibration_patches.py \
  --timing --crops .6 .75 --output /tmp/pixel_patch_60_timing.json
```

## Running the original comparison

From the repository root, using the existing project environment:

```sh
PYTEST_RUNNING=1 uv run python tests/benchmark_pixel_calibration_patches.py \
  --output /tmp/pixel_patch_results.jsonl

PYTEST_RUNNING=1 uv run python tests/benchmark_pixel_calibration_patches.py \
  --size 512 --samples 3 --repeats 2 \
  --scenes blurred sparse illumination --distances .12 \
  --output /tmp/pixel_patch_512.jsonl

PYTEST_RUNNING=1 uv run python tests/benchmark_pixel_calibration_patches.py \
  --timing --output /tmp/pixel_patch_timing.json
```

The first two commands produce per-run JSON records and print per-group pass
counts and maximum matrix errors. The third writes timing medians. Runtime
values vary by machine; fixed-input numerical results should remain close
across numerical-library implementations. The script is opt-in and is not
collected as part of the ordinary pytest suite.

This evaluation does not establish hardware accuracy or cover all samples.
It excludes camera quantization, saturation, shot-noise models, stage
backlash, optical distortion, drift, deformation, RGB-channel differences,
rectangular sensors, and safety-radius-limited travel. The production tests
cover some of these concerns separately; hardware validation remains needed.
