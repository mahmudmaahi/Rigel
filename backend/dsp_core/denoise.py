"""
dsp_core/denoise.py — Module 07: Noise Removal / Speech Enhancement
====================================================================

This module implements the classical offline denoising pipeline for Rigel.
It is strictly separate from the visualization STFT in stft.py.

The visualization STFT (stft.py) discards phase, normalises magnitudes to
dBFS, and downsamples axes for compact JSON payloads.  It cannot be used
for audio resynthesis.

This module provides:
    - A processing-grade STFT that preserves the full complex spectrum.
    - An ISTFT that reconstructs the time-domain signal via proper
      Overlap-Add (OLA) with element-wise window-sum normalisation.
    - Time-varying Minimum-Statistics noise estimation.
    - Improved power-domain Spectral Subtraction.
    - Decision-Directed Wiener Filtering (Ephraim-Malah style).

===========================================================================
CONVENTIONS AND BOUNDARY ASSUMPTIONS
===========================================================================

Frame-length (L), hop-length (H), window (w):
    Analysis and synthesis both use the same Hann window of length L.
    The hop length H = L // 4  (75 % overlap) unless overridden.
    A Hann window with 75 % overlap satisfies the Constant Overlap-Add
    (COLA) condition: the sum of squared windows accumulates to a constant
    1.5 per output sample (interior samples).  Boundary samples accumulate
    less; we handle this correctly via element-wise OLA normalisation.

Padding:
    Before framing, the signal is zero-padded by (L - H) samples at the
    start (centre-aligned convention) and by L samples at the end to ensure
    the final samples are covered by at least one complete frame.  The
    reconstructed signal is trimmed back to the original length after OLA.

One-sided spectrum:
    np.fft.rfft is used, returning L // 2 + 1 complex bins.  The zero-
    frequency (DC) bin and, for even L, the Nyquist bin are purely real.

Phase preservation:
    The denoising pipeline modifies only the magnitude of the complex STFT.
    The original complex phase ∠X(m, k) is preserved exactly and
    recombined with the enhanced magnitude after gain computation.

STFT → ISTFT round-trip reconstruction:
    For a signal x of any length (including lengths not divisible by H),
    the pipeline:
        stft_process(x, ...) → complex_stft
        istft_process(complex_stft, ..., length=len(x)) → y
    produces y ≈ x within floating-point precision.  This is tested with
    multiple signal lengths (exact multiples, off-by-one, primes, very
    short, long) in tests/test_denoise.py.  "Near-perfect" means
    max(|x - y|) < 1e-10 and np.allclose(x, y, atol=1e-7) under the
    assumptions above.

Stereo:
    Each channel is processed independently.  The same noise estimation
    logic is applied per channel.  This preserves mathematically exact
    per-channel reconstruction but does NOT guarantee that the stereo
    spatial image (inter-channel amplitude/phase relationship) is
    unchanged.  This tradeoff is documented in the test suite.

Amplitude scale:
    Internally all samples are converted to float64 in [-1, 1] before
    processing.  The reconstructed signal is returned as float32 in
    [-1, 1] for compatibility with the existing WAV encoding pipeline.

===========================================================================
ALGORITHM NOTES (Final Module 07 Architecture)
===========================================================================

Module 07 implements four progressive offline denoising approaches:

1. Improved Spectral Subtraction
   (denoise_spectral_subtraction)
2. Decision-Directed Wiener Filtering
   (denoise_wiener_dd)
3. Log-MMSE
   (denoise_logmmse)
4. IMCRA + OM-LSA
   (denoise_omlsa)

Shared Infrastructure:
    stft_process()    — analysis STFT (complex spectrum, no phase discard)
    istft_process()   — ISTFT via OLA with element-wise window normalisation
    _hann_window()    — shared Hann window helper
    estimate_noise_minima() — Phase 2 Minimum-Statistics noise tracking
    estimate_noise_imcra()  — Phase 6 IMCRA noise tracking

===========================================================================
ALGORITHM NOTES (Phase 2 — Minimum Statistics noise tracking)
===========================================================================

Phase 2 adds:
    estimate_noise_minima()  — time-varying noise PSD estimate P_noise(m, k)

Mathematical formulation
------------------------

1. Temporal smoothing of the noisy power spectrum:

       P_smooth(m, k) = alpha_s · P_smooth(m-1, k)
                      + (1 - alpha_s) · P_noisy(m, k)

   alpha_s in [0, 1) controls the decay rate.  Higher alpha_s = more
   smoothing = slower adaptation.  Default: 0.98.

   The IIR time constant (the lag at which the memory decays to 1/e) is:
       tau = -hop_length / (Fs * ln(alpha_s))
   For alpha_s=0.98, hop=512, Fs=44100:
       tau ≈ -512 / (44100 * ln(0.98)) ≈ 0.575 seconds.
   The 1.5-second figure refers to the default minimum-statistics look-back
   window (W * hop / Fs), not to the IIR smoother time constant.

   P_smooth estimates the local mean power with reduced variance.  It does
   NOT classify frames as speech or noise.  It is a statistical summary of
   the recent power history used to make the minimum search more stable.

2. Sliding-window minimum:

       P_min(m, k) = min_{i in [max(0, m-W+1), m]} P_smooth(i, k)

   W (window_frames) is the look-back window width in frames.  Default:
   covering approximately 1.5 seconds of audio.

   The idea: over a sufficiently long window, speech activity causes
   transient increases in P_smooth, while the background noise floor
   creates the sustained minimum.  P_min therefore approximates the noise
   floor in each bin, provided at least some frames in the window are
   noise-dominated (i.e., speech is absent or low-energy in that bin).

   This assumption can be violated:
     - If the window is entirely speech-dominated (e.g., a continuous
       vowel across all bins for > W frames), P_min over-estimates the
       noise floor in the affected bins.
     - For a rising noise floor, the temporal response depends on both
       the IIR smoother (which begins responding immediately via alpha_s)
       and the sliding minimum window (which cannot drop its minimum until
       W frames of high-power content have entered the buffer).  A step
       increase in noise power is therefore visible in P_smooth within
       a few IIR time constants (~0.575 s for the default alpha_s) but
       the minimum P_min only rises after the entire W-frame window has
       been filled with the new noise level.  For a step decrease, P_min
       drops quickly as soon as a single low-power frame enters the buffer.
     - Tonal noise (pure tones) and impulsive noise may be poorly
       estimated if the minimum window captures an impulsive peak.

   These are inherent limitations of pure Minimum Statistics.  More
   sophisticated estimators (e.g., MCRA with speech-presence probability)
   mitigate some of them at the cost of additional complexity.

3. Bias correction:

       P_noise(m, k) = B · P_min(m, k)

   The minimum of a smoothed power process is systematically lower than
   its true mean because the minimum operation selects the downward tail
   of the distribution.  A correction factor B compensates for this bias.
   Default: B = 1.5.

   B depends on alpha_s and W (and implicitly on signal statistics).  The
   default value is empirically established for typical speech signals and
   is not universally optimal.  For non-speech signals or unusual noise
   statistics the bias may differ.

4. Initialization:

   At m = 0, P_smooth is initialized to P_noisy[0] (the power of the
   first frame).  The minimum buffer is pre-filled with P_noisy[0] for
   all W slots.  This gives a reasonable starting estimate but means
   the tracker is biased toward the first-frame power during the initial
   W frames.  If the first frame is noisy, the early estimate is
   approximately correct.  If the first frame is dominated by speech,
   the initial noise estimate will be inflated and will only correct
   once quieter frames are observed within the window.

5. Computational complexity:

   O(n_frames × n_freq × W) in the naive implementation.
   For n_freq = 1025 (L=2048), W = 60 frames, n_frames = 500:
       ≈ 500 × 1025 × 60 = 30 750 000 scalar comparisons.
   This is entirely in NumPy (vectorised per-frame min over buffer)
   and is practical for offline processing.

6. Output:

   Returns P_noise of shape (n_frames, n_freq), dtype float64.
   All values are guaranteed finite and >= 0 (enforced with np.maximum
   after bias correction).
   P_noise(m, k) is an estimate of the noise power in bin k at frame m,
   not a label, threshold, or decision.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants (V1 defaults)
# ---------------------------------------------------------------------------

# Default STFT frame and hop length.
# At 44 100 Hz: L ≈ 46 ms per frame, H ≈ 11.6 ms hop (75 % overlap).
DEFAULT_FRAME_LENGTH: int = 2048
DEFAULT_HOP_LENGTH: int = DEFAULT_FRAME_LENGTH // 4   # 75 % overlap

# Minimum number of samples required before processing is attempted.
# Signals shorter than one frame are returned as-is after zero-padding.
MIN_SIGNAL_LENGTH: int = 1

# Tiny floor preventing log10(0) and division by zero in power calculations.
_POWER_FLOOR: float = 1e-12


# ---------------------------------------------------------------------------
# Internal window helper
# ---------------------------------------------------------------------------


def _hann_window(length: int) -> np.ndarray:
    """Return a Hann analysis/synthesis window of the given length as float64.

    Definition (symmetric):
        w[n] = 0.5 * (1 - cos(2π n / (L - 1))),  n = 0 … L-1

    This is the same window definition used by np.hanning().  Using L - 1
    in the denominator gives w[0] = w[L-1] = 0 exactly for any L, which
    tapers cleanly to zero at both frame boundaries.

    A Hann window with 75 % overlap (hop = L // 4) satisfies the
    Constant Overlap-Add (COLA) condition.  The accumulated analysis ×
    synthesis window product sums to 1.5 at interior samples.  Boundary
    samples accumulate less because fewer frames cover them; the OLA
    normalisation divides element-wise by this accumulated sum, so
    boundary samples are still correctly reconstructed.

    Parameters
    ----------
    length:
        Number of samples per frame (L).

    Returns
    -------
    float64 NumPy array of shape (L,).
    """
    if length < 1:
        raise ValueError(f"Window length must be >= 1; got {length}.")
    if length == 1:
        return np.ones(1, dtype=np.float64)
    return np.hanning(length).astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 1 — Processing STFT / ISTFT
# ---------------------------------------------------------------------------


def stft_process(
    signal: np.ndarray,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int | None = None,
) -> np.ndarray:
    """Compute the processing STFT of a single-channel signal.

    This function preserves the full complex spectrum for each frame so
    that the original phase can be combined with a modified magnitude during
    denoising.  It is intentionally separate from the visualization STFT
    in stft.py, which discards phase and downsamples the output.

    Parameters
    ----------
    signal:
        1-D NumPy array of float64 samples in [-1, 1].
        Must represent a single audio channel.
    frame_length:
        STFT frame length L in samples.  Controls frequency resolution:
            Δf = Fs / L  (Hz per bin)
        Default: 2048 samples.
    hop_length:
        Hop size H in samples (stride between successive frame starts).
        Default: frame_length // 4  (75 % overlap).

    Returns
    -------
    complex128 NumPy array of shape (n_frames, L // 2 + 1).
        stft[m, k] is the complex STFT coefficient at frame m, bin k.
        Magnitudes: |stft[m, k]|.  Phase: np.angle(stft[m, k]).

    Raises
    ------
    ValueError:
        If frame_length < 2 or hop_length < 1.

    Notes
    -----
    Padding convention:
        The signal is zero-padded by (L - H) samples at the start and by
        L samples at the end.  This centre-aligned convention ensures:
            - Frame 0 is centred near sample 0 of the original signal.
            - Every original sample is covered by at least one complete
              frame, including the very last samples.
        The ISTFT trims the reconstruction back to the original length.

    Frame extraction:
        Frame m starts at padded index m * H.
        Total number of frames: n_frames = 1 + (len(padded) - L) // H.

    Window:
        A Hann analysis window is applied to each frame before rfft.
        No amplitude normalisation is applied inside stft_process; the
        ISTFT uses the accumulated window-product sum for reconstruction.
    """
    if frame_length < 2:
        raise ValueError(f"frame_length must be >= 2; got {frame_length}.")

    if hop_length is None:
        hop_length = frame_length // 4

    if hop_length < 1:
        raise ValueError(f"hop_length must be >= 1; got {hop_length}.")

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(
            f"signal must be 1-D; got shape {signal.shape}. "
            "Call stft_process separately for each channel."
        )

    # Zero-pad: (L - H) at start, L at end.
    pad_start = frame_length - hop_length
    pad_end = frame_length
    padded = np.concatenate([
        np.zeros(pad_start, dtype=np.float64),
        signal,
        np.zeros(pad_end, dtype=np.float64),
    ])

    window = _hann_window(frame_length)
    n_padded = len(padded)
    n_frames = 1 + (n_padded - frame_length) // hop_length
    n_freq = frame_length // 2 + 1

    stft = np.zeros((n_frames, n_freq), dtype=np.complex128)

    for m in range(n_frames):
        start = m * hop_length
        frame = padded[start : start + frame_length]
        stft[m] = np.fft.rfft(frame * window)

    return stft


def get_first_valid_frame_index(frame_length: int, hop_length: int) -> int:
    """Return the index of the first STFT frame that contains no artificial zero-padding.
    
    The stft_process function prepends max(0, frame_length - hop_length) zeros 
    to the signal. Frame m starts at index m * hop_length in the padded array.
    To contain no padding, the frame must start at or after the padding ends.
    """
    pad_start = max(0, frame_length - hop_length)
    return (pad_start + hop_length - 1) // hop_length


def compute_robust_initial_noise(
    power_spectrum: np.ndarray,
    initial_frame: int = 0,
    percentile: float = 10.0
) -> np.ndarray:
    """Compute a robust offline initial noise spectrum from a noisy signal.
    
    This function analyzes the entire valid duration of the recording offline.
    It identifies a low-energy percentile of frames (by default, the lowest 10%)
    and computes their arithmetic mean to form a stable empirical baseline
    noise spectrum.
    
    This provides a robust initialization that avoids the causal cold-start 
    problem (where speech at the very beginning of a recording poisons the 
    noise estimator), while safely falling back to all valid frames if the 
    recording is extremely short. The 10% percentile is a reasonable empirical 
    default for the tested conditions.
    
    Parameters
    ----------
    power_spectrum:
        2D array of shape (n_frames, n_freq) containing the STFT power.
    initial_frame:
        The index of the first valid (unpadded) frame. Frames before this
        are excluded from the candidate pool.
    percentile:
        The percentage (0 to 100) of lowest-energy frames to select as
        candidate noise frames. Default is 10.0.
        
    Returns
    -------
    float64 array of shape (n_freq,) representing the robust initial
    noise power spectrum.
    """
    n_frames, n_freq = power_spectrum.shape
    initial_frame = max(0, min(initial_frame, n_frames - 1))
    
    valid_frames = power_spectrum[initial_frame:]
    n_valid = len(valid_frames)
    
    if n_valid == 0:
        return power_spectrum[initial_frame].copy()
        
    frame_energies = np.sum(valid_frames, axis=1)
    thresh = np.percentile(frame_energies, percentile)
    
    candidate_idx = np.where(frame_energies <= thresh)[0]
    
    if len(candidate_idx) == 0:
        candidate_idx = np.array([0])
        
    P_N_initial = np.mean(valid_frames[candidate_idx], axis=0)
    P_N_initial = np.maximum(P_N_initial, 1e-12)
    
    return P_N_initial


def istft_process(
    stft: np.ndarray,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int | None = None,
    original_length: int | None = None,
) -> np.ndarray:
    """Reconstruct a time-domain signal from a processing STFT via Overlap-Add.

    This function implements proper OLA with element-wise window-sum
    normalisation.  It handles boundary samples correctly by dividing each
    output sample by the exact accumulated analysis × synthesis window
    product at that position, rather than a single global constant.

    Parameters
    ----------
    stft:
        complex128 array of shape (n_frames, L // 2 + 1) as returned by
        stft_process.
    frame_length:
        STFT frame length L in samples.  Must match the value used in
        stft_process.
    hop_length:
        Hop size H.  Must match the value used in stft_process.
        Default: frame_length // 4.
    original_length:
        Length of the original signal before padding.  If provided, the
        reconstructed signal is trimmed (or zero-extended) to this length.
        If None, the full reconstructed buffer is returned.

    Returns
    -------
    float64 NumPy array of shape (original_length,) if original_length is
    given, else (reconstructed_length,).

    Raises
    ------
    ValueError:
        If frame_length < 2, hop_length < 1, or stft has unexpected shape.

    Notes
    -----
    Reconstruction algorithm:
        1. For each frame m, compute the inverse FFT:
               y_frame = np.fft.irfft(stft[m], n=L)
        2. Apply the synthesis Hann window to y_frame.
        3. Add y_frame into the output buffer at offset m * H.
        4. Accumulate the squared window product w_a[n] * w_s[n] into a
           normalisation buffer at the same offset.
        5. After all frames, divide the output buffer element-wise by the
           normalisation buffer, guarding against division by zero.
        6. Trim to original_length.

    Why element-wise normalisation:
        Interior samples are covered by exactly L // H = 4 overlapping
        frames (at 75 % overlap), and the squared Hann window sums to 1.5
        per sample.  Boundary samples are covered by fewer frames and
        accumulate proportionally less.  Dividing by the actual accumulated
        sum rather than a global constant gives correct reconstruction at
        every position, including boundaries.

    Perfect reconstruction:
        For an unmodified STFT (stft_process → istft_process with no gain
        applied), the reconstructed signal satisfies:
            max(|y - x|) < 1e-10   (typically ~1e-14)
            np.allclose(y, x, atol=1e-7)
        This is tested for multiple signal lengths in tests/test_denoise.py.
        The tolerance reflects accumulated floating-point arithmetic, not an
        algorithm defect.
    """
    if frame_length < 2:
        raise ValueError(f"frame_length must be >= 2; got {frame_length}.")

    if hop_length is None:
        hop_length = frame_length // 4

    if hop_length < 1:
        raise ValueError(f"hop_length must be >= 1; got {hop_length}.")

    if stft.ndim != 2:
        raise ValueError(
            f"stft must be 2-D (n_frames, n_freq); got shape {stft.shape}."
        )

    n_frames, n_freq = stft.shape
    expected_n_freq = frame_length // 2 + 1
    if n_freq != expected_n_freq:
        raise ValueError(
            f"stft frequency axis has {n_freq} bins but frame_length={frame_length} "
            f"requires {expected_n_freq} bins."
        )

    window = _hann_window(frame_length)
    # Analysis * synthesis window product (both Hann).
    win2 = window * window

    # Total reconstructed buffer length (matches the padded length from stft_process).
    total_length = (n_frames - 1) * hop_length + frame_length
    output = np.zeros(total_length, dtype=np.float64)
    norm = np.zeros(total_length, dtype=np.float64)

    for m in range(n_frames):
        # Inverse FFT; n=frame_length ensures we get L samples for even L.
        y_frame = np.fft.irfft(stft[m], n=frame_length)
        # Apply synthesis window.
        y_frame = y_frame * window
        start = m * hop_length
        output[start : start + frame_length] += y_frame
        norm[start : start + frame_length] += win2

    # Element-wise normalisation: guard against near-zero norm at edges.
    # Any position not covered by a single frame has norm == 0 (e.g., before
    # the first frame — not possible here, but guarded defensively).
    valid = norm > _POWER_FLOOR
    output[valid] /= norm[valid]

    # Trim or extend to original signal length.
    # The pad_start offset (frame_length - hop_length) applied in stft_process
    # shifts the original signal into the padded buffer; we reverse that here.
    pad_start = frame_length - hop_length
    if original_length is not None:
        result = output[pad_start : pad_start + original_length]
        if len(result) < original_length:
            result = np.pad(result, (0, original_length - len(result)))
    else:
        result = output[pad_start:]

    return result.astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 2 — Minimum Statistics noise tracking
# ---------------------------------------------------------------------------

# Minimum Statistics default parameters
# (documented fully in the module docstring above).
MS_DEFAULT_ALPHA_S: float = 0.98     # temporal smoothing coefficient
MS_DEFAULT_BIAS: float   = 1.5       # empirical bias correction factor
# Default look-back window expressed as a time span (seconds).
# Converted to frames at call time using the actual hop length / sample rate.
MS_DEFAULT_WINDOW_SECONDS: float = 1.5


def estimate_noise_minima(
    power_spectrum: np.ndarray,
    alpha_s: float = MS_DEFAULT_ALPHA_S,
    window_frames: int | None = None,
    bias: float = MS_DEFAULT_BIAS,
    sample_rate_hz: int = 44100,
    hop_length: int = DEFAULT_HOP_LENGTH,
    initial_frame: int = 0,
    P_N_initial: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the time-varying noise PSD using Minimum Statistics.

    For each STFT bin k and frame m, this function estimates P_noise(m, k),
    the background noise power spectral density, without requiring any prior
    knowledge of the noise spectrum and without assuming the first few frames
    are noise-only.

    The estimate is probabilistic and approximate.  It is not a binary
    speech/noise decision.  Its accuracy depends on the signal statistics,
    noise stationarity, SNR, and whether the minimum-statistics window
    contains any noise-dominated frames in each frequency bin.

    Parameters
    ----------
    power_spectrum:
        float64 array of shape (n_frames, n_freq) containing the noisy power
        spectrum at each frame.  Each element is |X(m, k)|^2 where X is the
        complex STFT coefficient.  Values must be finite and >= 0.
    alpha_s:
        Temporal smoothing coefficient in [0, 1).
        Governs the first-order IIR smoother applied to P_noisy before the
        minimum search:
            P_smooth(m, k) = alpha_s * P_smooth(m-1, k)
                           + (1 - alpha_s) * P_noisy(m, k)
        Higher alpha_s → more smoothing → slower adaptation.
        Default: 0.98 (≈ 50-frame / ~1.5 s time constant at Fs=44100,
        hop=512).  Must be in [0, 1).
    window_frames:
        Number of frames in the look-back minimum statistics window (W).
        If None, computed as:
            W = max(1, round(MS_DEFAULT_WINDOW_SECONDS * Fs / hop_length))
        using sample_rate_hz and hop_length.
        Larger W gives a more reliable minimum but lags the noise floor
        by up to W frames when the noise level rises.
    bias:
        Bias correction factor B (empirical, default 1.5).
        Applied as: P_noise(m, k) = B * P_min(m, k).
        The minimum of the smoothed power process is statistically lower
        than the true mean noise PSD. B is an empirical correction
        selected for the current estimator and evaluated scenarios to
        correct this downward bias.
        Must be >= 1.0.
    sample_rate_hz:
        Audio sample rate in Hz.  Used only to compute the default
        window_frames if window_frames is None.
    hop_length:
        STFT hop length in samples.  Used only to compute default
        window_frames if window_frames is None.

    Returns
    -------
    float64 NumPy array of shape (n_frames, n_freq).
    P_noise[m, k] is the estimated background noise power in frequency bin k
    at frame m.  All values are finite and >= 0.

    Raises
    ------
    ValueError:
        If power_spectrum is not 2-D, alpha_s not in [0, 1),
        bias < 1.0, or window_frames < 1.

    Notes
    -----
    Initialization (boundary behaviour, m < W):
        The minimum buffer is pre-filled with P_noisy[0] for all W slots.
        During the initial W frames the minimum is computed over fewer
        distinct observations; the estimate is dominated by the first-frame
        power.  If the recording begins with noise, this is approximately
        correct.  If it begins with speech, the early estimate will be
        inflated and will correct once noise-dominated frames enter the window.

    Limitations:
        - If an entire W-frame window is speech-dominated in some bin,
          P_noise in that bin is over-estimated.
        - For a rapidly rising noise floor, the estimate lags by up to W
          frames.  For a rapidly falling noise floor, the minimum drops
          quickly (not a practical problem for typical denoising scenarios).
        - The bias correction factor B = 1.5 is empirical and may not be
          optimal for all noise types or smoothing parameters.
        - Tonal or impulsive noise can cause local over- or under-estimation.
        - This is not MCRA (Minima Controlled Recursive Averaging) or
          IMCRA.  Those algorithms additionally estimate speech-presence
          probability to weight the smoothing coefficient.  This simpler
          estimator applies a fixed smoothing coefficient at every frame.

    Example (numerical diagnostic):
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> # 2 seconds of white noise at power sigma^2 = 0.01
        >>> sigma2 = 0.01
        >>> noise = rng.normal(0, sigma2**0.5, 44100 * 2).astype(np.float64)
        >>> from dsp_core.denoise import stft_process, estimate_noise_minima
        >>> S = stft_process(noise, frame_length=2048, hop_length=512)
        >>> P = np.abs(S) ** 2
        >>> P_noise = estimate_noise_minima(P, sample_rate_hz=44100,
        ...                                hop_length=512)
        >>> # After the initial window, the estimate should converge near
        >>> # sigma^2 / n_freq  (power is spread across bins).
        >>> print(f"Mean P_noise after init: {P_noise[60:].mean():.6f}")
        >>> print(f"Total P_noisy mean:       {P.mean():.6f}")
        >>> # P_noise per bin should be much smaller than total; their ratio
        >>> # reflects how well energy is distributed across frequency.
    """
    if power_spectrum.ndim != 2:
        raise ValueError(
            f"power_spectrum must be 2-D (n_frames, n_freq); "
            f"got shape {power_spectrum.shape}."
        )
    if not (0.0 <= alpha_s < 1.0):
        raise ValueError(f"alpha_s must be in [0, 1); got {alpha_s}.")
    if bias < 1.0:
        raise ValueError(f"bias must be >= 1.0; got {bias}.")

    power_spectrum = np.asarray(power_spectrum, dtype=np.float64)
    n_frames, n_freq = power_spectrum.shape

    # ── Compute default window size from time ──────────────────────────────
    if window_frames is None:
        window_frames = max(1, round(
            MS_DEFAULT_WINDOW_SECONDS * sample_rate_hz / hop_length
        ))

    if window_frames < 1:
        raise ValueError(f"window_frames must be >= 1; got {window_frames}.")

    # ── Allocate output ───────────────────────────────────────────────────
    P_noise = np.empty((n_frames, n_freq), dtype=np.float64)

    # ── Initialise ────────────────────────────────────────────────────────
    initial_frame = max(0, min(initial_frame, n_frames - 1))
    # P_smooth is set to the provided robust offline noise spectrum, 
    # or falls back to the power of the first fully valid frame.
    if P_N_initial is not None:
        P_smooth_cur = P_N_initial.copy()
    else:
        P_smooth_cur = power_spectrum[initial_frame].copy()

    # Circular buffer of shape (window_frames, n_freq) storing recent
    # values of P_smooth.  Index buf_ptr indicates the slot to overwrite next.
    buf = np.empty((window_frames, n_freq), dtype=np.float64)
    buf[:] = P_smooth_cur   # pre-fill entire buffer

    # ── Frame-by-frame update ─────────────────────────────────────────────
    for m in range(n_frames):
        if m <= initial_frame:
            # Frames up to the first valid frame do not update the state.
            # They use the initial fully-valid state for their noise estimate.
            buf_ptr = m % window_frames
            buf[buf_ptr] = P_smooth_cur
        else:
            # IIR temporal smoothing:
            #   P_smooth(m, k) = alpha_s * P_smooth(m-1, k)
            #                  + (1 - alpha_s) * P_noisy(m, k)
            P_smooth_cur = alpha_s * P_smooth_cur + (1.0 - alpha_s) * power_spectrum[m]

            # Write into circular buffer at the next slot.
            buf_ptr = m % window_frames
            buf[buf_ptr] = P_smooth_cur

        # Minimum over the current buffer contents.
        # For m < window_frames the buffer is partially pre-filled with
        # frame-0 power; the min is still taken over all W slots (some
        # of which contain the initialization value).
        P_min = buf.min(axis=0)

        # Bias correction and floor at zero.
        P_noise[m] = np.maximum(0.0, bias * P_min)

    # Enforce finiteness defensively (e.g., if input contained NaN).
    np.nan_to_num(P_noise, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return P_noise


# ---------------------------------------------------------------------------
# Phase 3 — Improved Power-Domain Spectral Subtraction
# ---------------------------------------------------------------------------
#
# Mathematical formulation
# ========================
#
# Given:
#     X(m, k)      — complex STFT of the noisy signal at frame m, bin k
#     P_noisy(m,k) = |X(m,k)|^2   — noisy power spectrum
#     P_noise(m,k)                 — noise PSD estimate from Phase 2
#
# Step 1 — Enhanced power (power domain, NOT magnitude domain):
#
#     P_enh(m,k) = max(
#         P_noisy(m,k) - alpha * P_noise(m,k),   ← noise subtraction
#         beta * P_noisy(m,k)                     ← spectral floor
#     )
#
#     alpha >= 1   : oversubtraction factor.  alpha=1 is exact subtraction.
#                   Larger alpha > 1 suppresses more residual noise and reduces
#                   gain-mask instability (musical noise) by burying noise peaks
#                   below the spectral floor, but increases signal distortion.
#                   Default: 1.0 (conservative).
#     beta in (0,1): spectral floor.  Ensures P_enh never drops to zero.
#
# Step 2 — Real-valued gain mask (maps to magnitude ratio):
#
#     G_raw(m,k) = sqrt(P_enh(m,k) / P_noisy(m,k))
#
# Step 3 — Temporal smoothing of gain mask:
#
#     G_t(m,k) = gamma_g * G_t(m-1,k) + (1 - gamma_g) * G_raw(m,k)
#
# Step 4 — Frequency smoothing of gain mask:
#
#     G_final(m,k) = UniformAvg over FREQ_SMOOTH_BINS neighboring bins.
#
# Known limitations:
#     1. Continuous signals: Minimum Statistics assumes the signal pauses
#        frequently enough for the tracker to find the true noise floor.
#        Continuous tones or unusually long speech without pauses will be
#        tracked as noise and severely attenuated.
#     2. Overlapping frequencies: Spectral subtraction uses a scalar gain
#        per bin. If signal and noise occupy the same time-frequency bin,
#        the signal component cannot be separated and will be attenuated.
#     3. Musical noise vs. Distortion: Low alpha preserves the signal but
#        leaves rapidly fluctuating residual noise peaks ("musical noise").
#        High alpha reduces these artifacts but attenuates the signal.
#        The default parameters represent a conservative trade-off.
#
# Stereo:
#     Processes channels independently, which may not preserve the exact
#     stereo spatial image.

from scipy.ndimage import uniform_filter1d   # noqa: E402 — local import for Phase 3

# Spectral subtraction parameter defaults
SS_DEFAULT_ALPHA: float = 1.0    # oversubtraction factor (>= 1.0)
SS_DEFAULT_BETA:  float = 0.01   # spectral floor (0 < beta < 1)

# Fixed internal smoothing defaults — not exposed as user parameters.
# These reduce musical noise without adding to the UI parameter space.
_GAIN_SMOOTH_ALPHA: float = 0.7  # temporal IIR smoothing coefficient for gain
_FREQ_SMOOTH_BINS: int   = 3     # uniform-average frequency smoothing width


def apply_spectral_subtraction(
    stft: np.ndarray,
    noise_psd: np.ndarray,
    alpha: float = SS_DEFAULT_ALPHA,
    beta:  float = SS_DEFAULT_BETA,
) -> np.ndarray:
    """Apply improved power-domain spectral subtraction to one STFT channel.

    This is the core Phase 3 function.  It operates entirely in the STFT
    domain and does NOT call stft_process or istft_process.  The caller is
    responsible for computing the STFT and reconstructing the waveform.

    Parameters
    ----------
    stft:
        complex128 array of shape (n_frames, n_freq) — the complex STFT of
        one audio channel, as returned by stft_process().
    noise_psd:
        float64 array of shape (n_frames, n_freq) — the time-varying noise
        power estimate P_noise(m, k), as returned by estimate_noise_minima().
        Must have the same shape as stft.
    alpha:
        Oversubtraction factor.  Must be >= 1.0.  Controls how aggressively
        the estimated noise PSD is subtracted.  alpha=1 is exact subtraction
        of the noise estimate.  alpha>1 subtracts more than the estimate,
        which helps when the noise estimate is conservative but can distort
        speech if the estimate is already accurate.
        Default: 2.0.
    beta:
        Spectral floor.  Must be in (0, 1).  The enhanced power is not
        allowed to drop below beta * P_noisy, preventing the gain from
        reaching zero.  This limits the maximum achievable attenuation.
        Smaller beta → more attenuation possible, higher musical-noise risk.
        Default: 0.01 (−20 dB floor).

    Returns
    -------
    complex128 NumPy array of shape (n_frames, n_freq) — the enhanced
    STFT Y(m, k) with modified magnitude and original phase.

    Raises
    ------
    ValueError:
        If stft and noise_psd shapes disagree, alpha < 1, or beta out of range.

    Notes
    -----
    Power-domain vs magnitude-domain:
        The subtraction is performed on power (P = |X|^2), not on magnitude
        (|X|).  Power-domain subtraction is consistent with a signal-plus-
        independent-noise model and avoids the underestimation artefact of
        magnitude-domain subtraction.

    Gain bounds:
        G is guaranteed to lie in [sqrt(beta), 1.0] per bin per frame for
        finite non-negative noise_psd values.  This is enforced analytically
        by the max() in the enhanced-power formula and the clip in the code.

    Smoothing (internal, fixed defaults):
        Temporal:  IIR with gamma_g = {_GAIN_SMOOTH_ALPHA} (initialised to G_raw[0]).
        Frequency: uniform average over {_FREQ_SMOOTH_BINS} consecutive bins per frame.
        Both steps are applied to the gain mask, not to the power spectrum,
        to avoid introducing amplitude artefacts at boundaries.
    """.format(_GAIN_SMOOTH_ALPHA=_GAIN_SMOOTH_ALPHA, _FREQ_SMOOTH_BINS=_FREQ_SMOOTH_BINS)

    if stft.ndim != 2 or noise_psd.ndim != 2:
        raise ValueError(
            "stft and noise_psd must be 2-D; "
            f"got stft {stft.shape}, noise_psd {noise_psd.shape}."
        )
    if stft.shape != noise_psd.shape:
        raise ValueError(
            f"stft and noise_psd must have the same shape; "
            f"got {stft.shape} vs {noise_psd.shape}."
        )
    if alpha < 1.0:
        raise ValueError(f"alpha must be >= 1.0; got {alpha}.")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1); got {beta}.")

    stft      = np.asarray(stft,      dtype=np.complex128)
    noise_psd = np.asarray(noise_psd, dtype=np.float64)

    n_frames, n_freq = stft.shape

    # Step 1 — Noisy power spectrum.
    P_noisy = np.abs(stft) ** 2   # shape: (n_frames, n_freq), float64

    # Step 2 — Enhanced power (power-domain subtraction with spectral floor).
    #
    #   P_enh(m,k) = max(P_noisy(m,k) - alpha*P_noise(m,k), beta*P_noisy(m,k))
    #
    # Using np.maximum for element-wise max (avoids Python loops).
    P_sub   = P_noisy - alpha * noise_psd          # may be negative
    P_floor = beta * P_noisy                        # spectral floor
    P_enh   = np.maximum(P_sub, P_floor)            # shape: (n_frames, n_freq)

    # Step 3 — Real-valued gain mask.
    #
    #   G_raw(m,k) = sqrt(P_enh(m,k) / P_noisy(m,k))
    #
    # Guard against division by zero in silent frames.
    P_noisy_safe = np.maximum(P_noisy, _POWER_FLOOR)
    G_raw = np.sqrt(P_enh / P_noisy_safe)          # in [sqrt(beta), 1.0]

    # Clip defensively to [sqrt(beta), 1.0] to absorb floating-point drift.
    G_raw = np.clip(G_raw, np.sqrt(beta), 1.0)

    # Step 4 — Temporal IIR smoothing of gain mask.
    #
    #   G_t(m,k) = gamma_g * G_t(m-1,k) + (1 - gamma_g) * G_raw(m,k)
    #
    G_t = np.empty_like(G_raw)
    G_t[0] = G_raw[0]
    for m in range(1, n_frames):
        G_t[m] = _GAIN_SMOOTH_ALPHA * G_t[m - 1] + (1.0 - _GAIN_SMOOTH_ALPHA) * G_raw[m]

    # Step 5 — Frequency smoothing of gain mask.
    #
    #   G_final(m,k) = uniform average over FREQ_SMOOTH_BINS neighbouring bins.
    #
    # uniform_filter1d operates along axis=1 (frequency), mode='nearest'
    # replicates edge bins rather than introducing boundary artefacts.
    G_final = uniform_filter1d(G_t, size=_FREQ_SMOOTH_BINS, axis=1, mode="nearest")

    # Final clip: smoothing cannot push G outside [sqrt(beta), 1.0] by more
    # than floating-point noise, but clip defensively.
    G_final = np.clip(G_final, np.sqrt(beta), 1.0)

    # Step 6 — Apply gain and preserve original phase.
    #
    #   Y(m,k) = G_final(m,k) * X(m,k)
    #
    # G_final is real, so angle(Y) = angle(X) exactly.
    enhanced_stft = G_final * stft   # complex128, same shape as input

    return enhanced_stft


def denoise_spectral_subtraction(
    signal: np.ndarray,
    sample_rate_hz: int = 44100,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int | None = None,
    alpha: float = SS_DEFAULT_ALPHA,
    beta:  float = SS_DEFAULT_BETA,
    noise_alpha_s: float = MS_DEFAULT_ALPHA_S,
    noise_window_frames: int | None = None,
    noise_bias: float = MS_DEFAULT_BIAS,
) -> np.ndarray:
    """Denoise an audio signal using improved power-domain spectral subtraction.

    Full single-channel or stereo pipeline:

        signal (1-D or 2-D)
            → normalize to float64 in [-1, 1]
            → per-channel STFT
            → per-channel Minimum Statistics noise estimate
            → per-channel spectral subtraction gain
            → per-channel ISTFT
            → output float32 in [-1, 1]

    Parameters
    ----------
    signal:
        NumPy array.  Mono: shape (N,).  Stereo: shape (N, 2).
        Integer or floating-point samples.  Will be normalized to float64
        in [-1, 1] before processing.
    sample_rate_hz:
        Audio sample rate in Hz.  Used to compute the default noise-tracker
        window length.
    frame_length:
        STFT frame length in samples.
    hop_length:
        STFT hop length.  Defaults to frame_length // 4 (75 % overlap).
    alpha:
        Oversubtraction factor (>= 1.0).  See apply_spectral_subtraction.
    beta:
        Spectral floor (0 < beta < 1).  See apply_spectral_subtraction.
    noise_alpha_s:
        IIR smoothing coefficient for the Minimum Statistics estimator.
    noise_window_frames:
        Look-back window for the noise estimator.  None → computed from
        MS_DEFAULT_WINDOW_SECONDS and the actual hop_length.
    noise_bias:
        Bias correction factor for the noise estimator.

    Returns
    -------
    float32 NumPy array of the same shape as signal.
    Values are in [-1, 1].  The output may clip slightly beyond ±1 if
    the noise estimator over-subtracts and reconstruction overshoots;
    downstream WAV encoding should apply a soft clip.

    Notes on stereo:
        Each channel is processed independently.  The noise estimate is
        computed separately per channel.  This preserves mathematically
        correct per-channel reconstruction but does NOT guarantee that the
        inter-channel amplitude or phase relationship (stereo spatial image)
        is preserved.  Asymmetric noise between channels is handled correctly
        (each channel attenuates its own noise independently).  For audio
        where stereo imaging is critical, the results should be evaluated
        perceptually.

    Notes on amplitude:
        The enhanced signal is returned in the same normalized float64 space
        as the input.  The ISTFT reconstructs into this space without any
        additional scaling.  The caller should convert to integer PCM for WAV
        export if required.
    """
    if hop_length is None:
        hop_length = frame_length // 4

    # Normalize to float64 in [-1, 1].
    from dsp_core.spectrum import normalize_samples   # noqa: PLC0415
    signal_f64 = normalize_samples(signal).astype(np.float64)

    mono = signal_f64.ndim == 1

    # Split into per-channel 1-D arrays.
    if mono:
        channels = [signal_f64]
    else:
        channels = [signal_f64[:, c] for c in range(signal_f64.shape[1])]

    n_samples = len(channels[0])
    enhanced_channels: list[np.ndarray] = []

    for ch_signal in channels:
        # 1. Analysis STFT (complex, full resolution).
        S = stft_process(ch_signal, frame_length=frame_length, hop_length=hop_length)

        # 2. Noisy power spectrum.
        P_noisy = np.abs(S) ** 2

        initial_frame = get_first_valid_frame_index(frame_length, hop_length)
        P_N_initial = compute_robust_initial_noise(P_noisy, initial_frame)

        # 3. Time-varying noise PSD estimate.
        P_noise = estimate_noise_minima(
            P_noisy,
            alpha_s=noise_alpha_s,
            window_frames=noise_window_frames,
            bias=noise_bias,
            sample_rate_hz=sample_rate_hz,
            hop_length=hop_length,
            initial_frame=initial_frame,
            P_N_initial=P_N_initial,
        )

        # 4. Spectral subtraction → enhanced complex STFT.
        S_enh = apply_spectral_subtraction(S, P_noise, alpha=alpha, beta=beta)

        # 5. Synthesis ISTFT (overlap-add, original length).
        y = istft_process(
            S_enh,
            frame_length=frame_length,
            hop_length=hop_length,
            original_length=n_samples,
        )
        enhanced_channels.append(y)

    # Reassemble channels.
    if mono:
        result = enhanced_channels[0]
    else:
        result = np.stack(enhanced_channels, axis=1)

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 5 — Log-MMSE Speech Enhancement
# ---------------------------------------------------------------------------

# Matches Spectral Subtraction's own default floor (sqrt(SS_DEFAULT_BETA) = 0.1).
# See denoise_logmmse's / denoise_wiener_dd's docstrings for why a floor is
# needed in practice.
LOGMMSE_DEFAULT_GMIN: float = 0.1


def apply_logmmse_filter(
    stft: np.ndarray,
    noise_psd: np.ndarray,
    alpha_dd: float = 0.98,
    g_min: float = 0.0,
) -> np.ndarray:
    """Apply the Log-MMSE (Ephraim & Malah 1985) estimator to a noisy STFT.

    Parameters
    ----------
    stft:
        Complex 2-D array (n_frames, n_freq) of the noisy signal.
    noise_psd:
        Real 2-D array (n_frames, n_freq) of the estimated noise power spectrum.
    alpha_dd:
        Decision-Directed smoothing parameter in [0, 1].
    g_min:
        Minimum gain floor in [0, 1]. Default 0.0 (no floor — matches the
        original formulation). The pipeline entry point `denoise_logmmse`
        uses a nonzero default in practice; see its docstring for why.

    Returns
    -------
    enhanced_stft:
        Complex 2-D array (n_frames, n_freq) after applying the Log-MMSE gain.
    """
    from scipy.special import exp1  # Ensure scipy is available for E1(v)

    if stft.ndim != 2 or noise_psd.ndim != 2:
        raise ValueError(
            f"stft and noise_psd must be 2-D; got {stft.shape}, {noise_psd.shape}."
        )
    if stft.shape != noise_psd.shape:
        raise ValueError(
            f"stft and noise_psd shape mismatch; got {stft.shape} vs {noise_psd.shape}."
        )
    if not (0.0 <= alpha_dd <= 1.0):
        raise ValueError(f"alpha_dd must be in [0, 1]; got {alpha_dd}.")

    stft = np.asarray(stft, dtype=np.complex128)
    noise_psd = np.asarray(noise_psd, dtype=np.float64)

    n_frames, n_freq = stft.shape
    epsilon = 1e-20

    # 1. Noisy power
    P_noisy = np.abs(stft) ** 2
    
    # 2. A-posteriori SNR (gamma)
    P_noise_safe = np.maximum(noise_psd, epsilon)
    gamma = P_noisy / P_noise_safe
    
    # 3. Decision-Directed A-priori SNR (xi)
    xi = np.zeros_like(P_noisy, dtype=np.float64)
    
    # Pre-allocate output
    enhanced_stft = np.zeros_like(stft, dtype=np.complex128)
    
    EULER_GAMMA = 0.57721566490153286
    V_THRESHOLD = 1e-3
    
    # Helper function to compute G_LMMSE safely for a single frame
    def compute_gain(xi_m, gamma_m, X_m):
        G_m = np.zeros_like(xi_m)
        
        # We only compute gain where |X| > 0.
        # Where |X| == 0, the theoretical gain might diverge if xi > 0,
        # but the product G * |X| is finite. However, phase is undefined,
        # and numerical Inf * 0 is NaN. We strictly bypass these to force Y=0.
        valid_mask = (P_noisy[m] > 0)
        
        # Further split valid_mask into small v and large v branches
        xi_valid = xi_m[valid_mask]
        gamma_valid = gamma_m[valid_mask]
        
        # v = [xi / (1 + xi)] * gamma
        v_valid = (xi_valid / (1.0 + xi_valid)) * gamma_valid
        
        G_valid = np.zeros_like(v_valid)
        
        # Branch 1: v <= V_THRESHOLD
        # Use first-order asymptotic expansion: E1(v) ≈ -gamma_E - ln(v) + v
        # G ≈ sqrt(xi / (1 + xi)) * exp(-0.5 * gamma_E) / sqrt(gamma) * exp(0.5 * v)
        small_mask = (v_valid <= V_THRESHOLD)
        if np.any(small_mask):
            xi_small = xi_valid[small_mask]
            gamma_small = gamma_valid[small_mask]
            v_small = v_valid[small_mask]
            
            # If xi == 0, G is strictly 0.
            # Handle xi > 0 safely.
            G_small = np.zeros_like(xi_small)
            nonzero_xi = (xi_small > 0)
            
            if np.any(nonzero_xi):
                xi_nz = xi_small[nonzero_xi]
                gamma_nz = gamma_small[nonzero_xi]
                v_nz = v_small[nonzero_xi]
                
                term1 = np.sqrt(xi_nz / (1.0 + xi_nz))
                term2 = np.exp(-0.5 * EULER_GAMMA) / np.sqrt(gamma_nz)
                term3 = np.exp(0.5 * v_nz)
                G_small[nonzero_xi] = term1 * term2 * term3
                
            G_valid[small_mask] = G_small
            
        # Branch 2: v > V_THRESHOLD
        # Evaluate exact formulation using scipy.special.exp1
        large_mask = ~small_mask
        if np.any(large_mask):
            xi_large = xi_valid[large_mask]
            v_large = v_valid[large_mask]
            
            wiener_factor = xi_large / (1.0 + xi_large)
            exp_integral = exp1(v_large)
            # Clip exp_integral to prevent extremely rare overflow if v is just above threshold
            # (though exp1(1e-3) is approx 6.3, exp(3.15) is fine).
            correction = np.exp(0.5 * exp_integral)
            
            G_valid[large_mask] = wiener_factor * correction
            
        G_m[valid_mask] = G_valid
        return G_m

    if n_frames > 0:
        # Frame 0 initialization
        xi[0] = np.maximum(gamma[0] - 1.0, 0.0)
        m = 0
        G_0 = compute_gain(xi[0], gamma[0], stft[0])
        # Frequency-domain gain smoothing (same 3-bin uniform kernel used by
        # Spectral Subtraction) — suppresses isolated per-bin gain spikes
        # (e.g. from a frame whose window straddles a noise/speech boundary)
        # that would otherwise reconstruct as a narrowband, noise-phase
        # "musical noise" artifact. See apply_wiener_filter_dd for the full note.
        G_0 = uniform_filter1d(G_0, size=_FREQ_SMOOTH_BINS, mode="nearest")
        if g_min > 0.0:
            G_0 = np.maximum(G_0, g_min)

        # Explicit bypass for |X|=0: where |X|=0, enhanced_stft is exactly 0 (already allocated).
        # We only apply gain where X != 0 to strictly avoid NaN propagation.
        valid_0 = (P_noisy[0] > 0)
        enhanced_stft[0, valid_0] = G_0[valid_0] * stft[0, valid_0]

        # Frame 1..N
        for m in range(1, n_frames):
            P_enh_prev = np.abs(enhanced_stft[m - 1]) ** 2

            # xi(m,k) DD update
            term1 = alpha_dd * (P_enh_prev / P_noise_safe[m])
            term2 = (1.0 - alpha_dd) * np.maximum(gamma[m] - 1.0, 0.0)

            xi[m] = np.maximum(term1 + term2, 0.0)

            G_m = compute_gain(xi[m], gamma[m], stft[m])
            G_m = uniform_filter1d(G_m, size=_FREQ_SMOOTH_BINS, mode="nearest")

            # Gain floor (g_min, off by default) — see
            # apply_wiener_filter_dd's docstring for the full rationale:
            # bounds how far gain can drop during noise-only frames, which
            # shrinks the noise-to-speech gain swing at a boundary frame and
            # so reduces the audible "pre-echo" smear from overlap-add,
            # without slowing the response to genuine, sustained speech.
            if g_min > 0.0:
                G_m = np.maximum(G_m, g_min)

            valid_m = (P_noisy[m] > 0)
            enhanced_stft[m, valid_m] = G_m[valid_m] * stft[m, valid_m]
            
    return enhanced_stft


def denoise_logmmse(
    signal: np.ndarray,
    sample_rate_hz: int = 44100,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int | None = None,
    alpha_dd: float = 0.98,
    noise_alpha_s: float = MS_DEFAULT_ALPHA_S,
    noise_window_frames: int | None = None,
    noise_bias: float = MS_DEFAULT_BIAS,
    g_min: float = LOGMMSE_DEFAULT_GMIN,
) -> np.ndarray:
    """Denoise an audio signal using Log-MMSE Speech Enhancement.

    This function routes mono or stereo signals through STFT, Phase 2 Minimum Statistics
    noise tracking, Log-MMSE filtering, and ISTFT reconstruction.

    g_min:
        Minimum gain floor in [0, 1]. Defaults to `LOGMMSE_DEFAULT_GMIN`
        (unlike `apply_logmmse_filter`'s own default of 0.0) for the same
        reason as `denoise_wiener_dd` — see its docstring.
    """
    if hop_length is None:
        hop_length = frame_length // 4

    from dsp_core.spectrum import normalize_samples
    signal_f64 = normalize_samples(signal).astype(np.float64)

    mono = signal_f64.ndim == 1
    channels = [signal_f64] if mono else [signal_f64[:, c] for c in range(signal_f64.shape[1])]
    
    n_samples = len(channels[0])
    enhanced_channels: list[np.ndarray] = []

    for ch_signal in channels:
        # 1. STFT
        S = stft_process(ch_signal, frame_length=frame_length, hop_length=hop_length)
        
        # 2. Noise estimation
        P_noisy = np.abs(S) ** 2
        
        initial_frame = get_first_valid_frame_index(frame_length, hop_length)
        P_N_initial = compute_robust_initial_noise(P_noisy, initial_frame)
        
        P_noise = estimate_noise_minima(
            P_noisy,
            alpha_s=noise_alpha_s,
            window_frames=noise_window_frames,
            bias=noise_bias,
            sample_rate_hz=sample_rate_hz,
            hop_length=hop_length,
            initial_frame=initial_frame,
            P_N_initial=P_N_initial,
        )
        
        # 3. Log-MMSE filter
        S_enh = apply_logmmse_filter(S, P_noise, alpha_dd=alpha_dd, g_min=g_min)
        
        # 4. ISTFT
        y = istft_process(
            S_enh,
            frame_length=frame_length,
            hop_length=hop_length,
            original_length=n_samples,
        )
        enhanced_channels.append(y)

    result = enhanced_channels[0] if mono else np.stack(enhanced_channels, axis=1)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 4 — Decision-Directed Wiener Filtering
# ---------------------------------------------------------------------------

# Matches Spectral Subtraction's own default floor (sqrt(SS_DEFAULT_BETA) = 0.1).
# See denoise_wiener_dd's docstring for why a floor is needed in practice.
WIENER_DEFAULT_GMIN: float = 0.1


def apply_wiener_filter_dd(
    stft: np.ndarray,
    noise_psd: np.ndarray,
    alpha_dd: float = 0.98,
    g_min: float = 0.0,
) -> np.ndarray:
    """Apply a Decision-Directed Wiener filter to a noisy STFT.

    Parameters
    ----------
    stft:
        Complex 2-D array (n_frames, n_freq) of the noisy signal.
    noise_psd:
        Real 2-D array (n_frames, n_freq) of the estimated noise power spectrum.
    alpha_dd:
        Decision-Directed smoothing parameter in [0, 1].
        Typically 0.98. Controls the weighting of the a-priori SNR estimate
        from the previous enhanced frame vs the current a-posteriori SNR.
    g_min:
        Minimum gain floor in [0, 1]. Default 0.0 (no floor — matches the
        original, purely mathematical Decision-Directed formulation, where
        gain is bounded in [0, 1] only by construction). The pipeline entry
        point `denoise_wiener_dd` uses a nonzero default in practice; see its
        docstring for why.

    Returns
    -------
    enhanced_stft:
        Complex 2-D array (n_frames, n_freq) after applying the Wiener gain.
        The phase is preserved exactly from the input.
    """
    if stft.ndim != 2 or noise_psd.ndim != 2:
        raise ValueError(
            f"stft and noise_psd must be 2-D; got {stft.shape}, {noise_psd.shape}."
        )
    if stft.shape != noise_psd.shape:
        raise ValueError(
            f"stft and noise_psd shape mismatch; got {stft.shape} vs {noise_psd.shape}."
        )
    if not (0.0 <= alpha_dd <= 1.0):
        raise ValueError(f"alpha_dd must be in [0, 1]; got {alpha_dd}.")

    stft = np.asarray(stft, dtype=np.complex128)
    noise_psd = np.asarray(noise_psd, dtype=np.float64)

    n_frames, n_freq = stft.shape
    epsilon = 1e-20  # Numerical safety against division by zero

    # 1. Noisy power
    P_noisy = np.abs(stft) ** 2
    
    # 2. A-posteriori SNR (gamma)
    P_noise_safe = np.maximum(noise_psd, epsilon)
    gamma = P_noisy / P_noise_safe
    
    # 3. Decision-Directed A-priori SNR (xi)
    xi = np.zeros_like(P_noisy, dtype=np.float64)
    
    # Pre-allocate output
    enhanced_stft = np.zeros_like(stft, dtype=np.complex128)
    
    if n_frames > 0:
        # Initialize first frame (no previous enhanced frame exists)
        # xi(0,k) = max(gamma(0,k) - 1, 0)
        xi[0] = np.maximum(gamma[0] - 1.0, 0.0)
        
        # First frame gain and application.
        #
        # Frequency-domain gain smoothing (same 3-bin uniform kernel used by
        # Spectral Subtraction, see apply_spectral_subtraction) is applied here
        # too: a frame whose analysis window straddles the onset of a new
        # sound (e.g. the boundary between a noise-only prefix and speech)
        # can see gamma/xi spike in a handful of isolated bins while the
        # frame's phase is still overwhelmingly noise-derived. Without
        # smoothing, that isolated per-bin gain boost is applied to
        # noise-phase content and reconstructs as a short, narrowband,
        # incoherent-sounding ("musical noise") artifact. Averaging the gain
        # over neighbouring bins suppresses isolated spikes while leaving
        # genuine broadband gain changes (real speech, many bins wide)
        # essentially unaffected.
        G_0 = xi[0] / (1.0 + xi[0])
        G_0 = uniform_filter1d(G_0, size=_FREQ_SMOOTH_BINS, mode="nearest")
        if g_min > 0.0:
            G_0 = np.maximum(G_0, g_min)
        enhanced_stft[0] = G_0 * stft[0]

        # Process remaining frames
        for m in range(1, n_frames):
            # The critical DD recursion using the PREVIOUS ENHANCED power: |Y(m-1, k)|^2
            # Explicitly implemented to satisfy the Phase 4 requirement.
            P_enh_prev = np.abs(enhanced_stft[m - 1]) ** 2

            # xi(m,k) = alpha_dd * [ |Y(m-1,k)|^2 / P_N(m,k) ] + (1 - alpha_dd) * max(gamma(m,k) - 1, 0)
            term1 = alpha_dd * (P_enh_prev / P_noise_safe[m])
            term2 = (1.0 - alpha_dd) * np.maximum(gamma[m] - 1.0, 0.0)

            xi[m] = term1 + term2

            # Ensure xi >= 0 (mathematically guaranteed by the max() and absolute value, but safe to clamp)
            xi[m] = np.maximum(xi[m], 0.0)

            # Wiener gain: G = xi / (1 + xi)
            # Because xi >= 0, this strictly bounds G in [0, 1]
            G_m = xi[m] / (1.0 + xi[m])

            # Frequency-domain gain smoothing — see note on G_0 above.
            G_m = uniform_filter1d(G_m, size=_FREQ_SMOOTH_BINS, mode="nearest")

            # Gain floor (g_min, off by default — see apply_wiener_filter_dd's
            # docstring for the full rationale). Unlike Spectral Subtraction
            # and OM-LSA, the plain Wiener/DD gain has no floor at all, so it
            # can swing between ~0 (steady noise) and ~1 (speech) with no
            # limit on how large that swing is. A frame whose analysis window
            # straddles a noise/speech boundary legitimately needs a higher
            # gain once real speech enters it — but that frame's synthesis
            # window spans both sides of the boundary, so overlap-add smears
            # part of its reconstruction into samples that are, in real time,
            # still before the audible onset ("pre-echo", a structural
            # consequence of applying one gain per analysis frame). Raising
            # the noise-floor gain shrinks that swing, so the smeared
            # contribution is far less audible, without slowing the response
            # to genuine, sustained speech.
            if g_min > 0.0:
                G_m = np.maximum(G_m, g_min)

            # Apply gain
            enhanced_stft[m] = G_m * stft[m]
        
    return enhanced_stft


def denoise_wiener_dd(
    signal: np.ndarray,
    sample_rate_hz: int = 44100,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int | None = None,
    alpha_dd: float = 0.98,
    noise_alpha_s: float = MS_DEFAULT_ALPHA_S,
    noise_window_frames: int | None = None,
    noise_bias: float = MS_DEFAULT_BIAS,
    g_min: float = WIENER_DEFAULT_GMIN,
) -> np.ndarray:
    """Denoise an audio signal using Decision-Directed Wiener Filtering.

    This function routes mono or stereo signals through STFT, Phase 2 Minimum Statistics
    noise tracking, Decision-Directed Wiener filtering, and ISTFT reconstruction.

    g_min:
        Minimum gain floor in [0, 1]. Defaults to `WIENER_DEFAULT_GMIN` (unlike
        `apply_wiener_filter_dd`'s own default of 0.0) because a floor is
        needed in practice: without one, a frame whose analysis window
        straddles a noise/speech boundary can legitimately swing from
        near-zero gain to a much higher gain the instant real speech enters
        its window, and that swing gets smeared backward in time by
        overlap-add ("pre-echo"), audible as a brief artifact immediately
        before the perceived onset. Raising the floor shrinks the swing
        without slowing the response to genuine, sustained speech, which
        spans many frames rather than one.
    """
    if hop_length is None:
        hop_length = frame_length // 4

    from dsp_core.spectrum import normalize_samples
    signal_f64 = normalize_samples(signal).astype(np.float64)

    mono = signal_f64.ndim == 1
    channels = [signal_f64] if mono else [signal_f64[:, c] for c in range(signal_f64.shape[1])]
    
    n_samples = len(channels[0])
    enhanced_channels: list[np.ndarray] = []

    for ch_signal in channels:
        # 1. STFT
        S = stft_process(ch_signal, frame_length=frame_length, hop_length=hop_length)
        
        # 2. Noise estimation
        P_noisy = np.abs(S) ** 2
        
        initial_frame = get_first_valid_frame_index(frame_length, hop_length)
        P_N_initial = compute_robust_initial_noise(P_noisy, initial_frame)
        
        P_noise = estimate_noise_minima(
            P_noisy,
            alpha_s=noise_alpha_s,
            window_frames=noise_window_frames,
            bias=noise_bias,
            sample_rate_hz=sample_rate_hz,
            hop_length=hop_length,
            initial_frame=initial_frame,
            P_N_initial=P_N_initial,
        )
        
        # 3. Wiener DD
        S_enh = apply_wiener_filter_dd(S, P_noise, alpha_dd=alpha_dd, g_min=g_min)
        
        # 4. ISTFT
        y = istft_process(
            S_enh,
            frame_length=frame_length,
            hop_length=hop_length,
            original_length=n_samples,
        )
        enhanced_channels.append(y)

    result = enhanced_channels[0] if mono else np.stack(enhanced_channels, axis=1)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 6A — IMCRA Noise Estimator
# ---------------------------------------------------------------------------

def estimate_noise_imcra(
    P_noisy: np.ndarray,
    alpha_s: float = 0.86,
    alpha_d: float = 0.85,
    w_freq: int = 1,
    U: int = 8,
    V_sub: int = 16,
    gamma0: float = 4.6,
    zeta0: float = 1.67,
    zeta_max: float = 10.0,
    zeta_min: float = 1.0,
    q_max: float = 0.998,
    beta_min: float = 1.47,
    alpha_dd: float = 0.98,
    initial_frame: int = 0,
    P_N_initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Improved Minima Controlled Recursive Averaging (IMCRA) noise estimator.
    
    Implementation of Cohen (2003) two-stage noise tracker, using Speech-Presence 
    Probability (SPP) to selectively update the noise PSD.
    
    Parameters
    ----------
    P_noisy:
        Real 2-D array (n_frames, n_freq) of the noisy STFT power.
    alpha_s:
        Time smoothing constant (0.86 corresponds to ~76 ms at 44.1kHz / 512 hop).
    alpha_d:
        Base noise update constant.
    w_freq:
        Frequency smoothing window (1 means a 3-bin normalized window).
    U, V_sub:
        Sub-window dimensions for minimum tracking (U=8, V_sub=16 -> 128 frames).
    gamma0, zeta0:
        Rough decision thresholds for a-posteriori and smoothed a-posteriori SNR.
    zeta_max, zeta_min:
        Interpolation limits for a-priori speech-absence probability q(m,k).
    q_max:
        Maximum allowed a-priori speech-absence probability.
    beta_min:
        Bias compensation factor. (1.47 is a literature-derived approx for V=128).
    alpha_dd:
        Decision-Directed parameter for the internal a-priori SNR (xi) tracker.
        
    Returns
    -------
    P_noise:
        Real 2-D array (n_frames, n_freq) of the estimated noise PSD.
    spp:
        Real 2-D array (n_frames, n_freq) of the Speech-Presence Probability p(m,k) in [0,1].
    """
    if P_noisy.ndim != 2:
        raise ValueError(f"P_noisy must be 2-D; got {P_noisy.shape}.")
        
    n_frames, n_freq = P_noisy.shape
    P_noise = np.zeros_like(P_noisy, dtype=np.float64)
    spp_out = np.zeros_like(P_noisy, dtype=np.float64)
    
    if n_frames == 0:
        return P_noise, spp_out
        
    epsilon = 1e-20
    
    # Frequency smoothing window (normalized)
    if w_freq == 1:
        b = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    else:
        # Fallback to no frequency smoothing if w_freq is 0
        b = np.array([1.0], dtype=np.float64)
        w_freq = 0

    # State variables initialization from the robust offline noise spectrum
    # (or the first valid frame as fallback)
    initial_frame = max(0, min(initial_frame, n_frames - 1))
    
    if P_N_initial is not None:
        P_init = P_N_initial.copy()
    else:
        P_init = P_noisy[initial_frame]
    
    if w_freq > 0:
        P_init_pad = np.pad(P_init, (w_freq, w_freq), mode='reflect')
        S_init = np.convolve(P_init_pad, b, mode='valid')
    else:
        S_init = P_init.copy()
        
    S_f = S_init.copy()
    S_f_tilde = S_init.copy()
    P_N = P_init.copy()
    xi = np.zeros(n_freq, dtype=np.float64)
    
    # Sub-window minimum tracking buffers
    S_min_sw = np.zeros((U, n_freq), dtype=np.float64)
    S_tilde_min_sw = np.zeros((U, n_freq), dtype=np.float64)
    S_min_sw[:] = S_f
    S_tilde_min_sw[:] = S_f_tilde
    
    S_min_running = S_f.copy()
    S_tilde_min_running = S_f_tilde.copy()
    
    sw_index = 0
    
    for m in range(n_frames):
        P_m = P_noisy[m]
        
        # 1. Frequency smoothing
        if w_freq > 0:
            P_padded = np.pad(P_m, (w_freq, w_freq), mode='reflect')
            S = np.convolve(P_padded, b, mode='valid')
        else:
            S = P_m.copy()
            
        # 2. Time smoothing
        if m <= initial_frame:
            # Frames before the first valid frame keep the initialized state
            S_f = S_init.copy()
            S_f_tilde = S_init.copy()
            S_min_running = S_f.copy()
            S_tilde_min_running = S_f_tilde.copy()
            P_N = P_init.copy()
            xi = np.maximum(P_m / (P_N + epsilon) - 1.0, 0.0)
        else:
            S_f = alpha_s * S_f + (1.0 - alpha_s) * S
            
        # Update running minimums
        np.minimum(S_min_running, S_f, out=S_min_running)
        
        # Global minimum (First iteration)
        # min over all completed sub-windows and the current running sub-window
        S_min_past = np.min(S_min_sw, axis=0)
        S_min = np.minimum(S_min_past, S_min_running)
        
        # 3. Rough speech-presence decision
        # beta_min corrects the bias of the tracked minimum
        gamma_min = P_m / (beta_min * S_min + epsilon)
        zeta = S_f / (beta_min * S_min + epsilon)
        
        I_indicator = np.zeros(n_freq, dtype=int)
        I_indicator[(gamma_min > gamma0) & (zeta > zeta0)] = 1
        
        # 4. Second smoothing iteration (excluding strong speech)
        alpha_c = np.where(I_indicator == 1, 1.0, alpha_s)
        if m > initial_frame:
            S_f_tilde = alpha_c * S_f_tilde + (1.0 - alpha_c) * S
            
        np.minimum(S_tilde_min_running, S_f_tilde, out=S_tilde_min_running)
        
        # 5. Second minimum tracking
        S_tilde_min_past = np.min(S_tilde_min_sw, axis=0)
        S_tilde_min = np.minimum(S_tilde_min_past, S_tilde_min_running)
        
        # 6. A-priori speech-absence probability q(m,k)
        zeta_tilde = S_f / (beta_min * S_tilde_min + epsilon)
        
        # Soft thresholding interpolation
        q = (zeta_max - zeta_tilde) / (zeta_max - zeta_min + epsilon)
        q = np.clip(q, 0.0, 1.0)
        q = np.minimum(q, q_max)
        
        # 7. Internal DD SNR tracking for SPP calculation
        gamma = P_m / (P_N + epsilon)
        if m > initial_frame:
            # We use an internal DD formulation decoupled from OM-LSA enhancement output
            xi = alpha_dd * xi + (1.0 - alpha_dd) * np.maximum(gamma - 1.0, 0.0)
            
        v = (xi / (1.0 + xi)) * gamma
        
        # 8. SPP p(m,k)
        p = np.zeros(n_freq, dtype=np.float64)
        
        # Prevent overflow in exp(v). If v >= 50, likelihood is huge, p -> 1.0
        safe_mask = (v < 50.0)
        
        q_safe = q[safe_mask]
        v_safe = v[safe_mask]
        xi_safe = xi[safe_mask]
        
        Lambda = ((1.0 - q_safe) / (q_safe + epsilon)) * (np.exp(v_safe) / (1.0 + xi_safe))
        p[safe_mask] = Lambda / (1.0 + Lambda)
        p[~safe_mask] = 1.0
        
        # 9. Time-Varying Recursive Noise Averaging
        alpha_d_tilde = alpha_d + (1.0 - alpha_d) * p
        if m > initial_frame:
            # IMCRA strictly updates with the raw observation P_m.
            # No bias compensation is applied here because P_m is an unbiased
            # estimate of the noise variance during speech absence (p -> 0).
            P_N = alpha_d_tilde * P_N + (1.0 - alpha_d_tilde) * P_m
            
        P_noise[m] = P_N.copy()
        
        # 10. Sub-window buffer shift
        if (m + 1) % V_sub == 0:
            S_min_sw[sw_index] = S_min_running
            S_tilde_min_sw[sw_index] = S_tilde_min_running
            
            sw_index = (sw_index + 1) % U
            
            S_min_running.fill(np.inf)
            S_tilde_min_running.fill(np.inf)
            
        spp_out[m] = p.copy()
            
    return P_noise, spp_out


# ---------------------------------------------------------------------------
# Phase 6B — OM-LSA Gain and Integration
# ---------------------------------------------------------------------------

# See denoise_omlsa's docstring for why the pipeline entry point raises this
# above apply_omlsa_filter's own default (0.01).
OMLSA_DEFAULT_GMIN: float = 0.05


def apply_omlsa_filter(
    stft: np.ndarray,
    noise_psd: np.ndarray,
    spp: np.ndarray,
    alpha_dd: float = 0.98,
    G_min: float = 0.01,
) -> np.ndarray:
    """Apply the Optimally Modified Log-Spectral Amplitude (OM-LSA) gain.
    
    Uses the Speech-Presence Probability (SPP) to geometrically modify the 
    Log-MMSE gain towards a subjective minimum gain (G_min) during speech absence.
    
    Parameters
    ----------
    stft:
        Complex 2-D array (n_frames, n_freq) of the noisy signal.
    noise_psd:
        Real 2-D array (n_frames, n_freq) of the estimated noise power spectrum.
    spp:
        Real 2-D array (n_frames, n_freq) of the estimated Speech-Presence Probability p(m,k).
    alpha_dd:
        Decision-Directed smoothing parameter in [0, 1] for a-priori SNR.
    G_min:
        Subjective minimum gain applied when speech is strictly absent (e.g., 0.01 = -40 dB).
        
    Returns
    -------
    enhanced_stft:
        Complex 2-D array (n_frames, n_freq) after applying the OM-LSA gain.
    """
    if stft.ndim != 2 or noise_psd.ndim != 2 or spp.ndim != 2:
        raise ValueError("stft, noise_psd, and spp must be 2-D.")
    if stft.shape != noise_psd.shape or stft.shape != spp.shape:
        raise ValueError("Shape mismatch between stft, noise_psd, and spp.")
        
    stft = np.asarray(stft, dtype=np.complex128)
    noise_psd = np.asarray(noise_psd, dtype=np.float64)
    spp = np.asarray(spp, dtype=np.float64)
    
    n_frames, n_freq = stft.shape
    epsilon = 1e-20
    
    P_noisy = np.abs(stft)**2
    P_noise_safe = np.maximum(noise_psd, epsilon)
    gamma = P_noisy / P_noise_safe
    
    xi = np.zeros_like(P_noisy, dtype=np.float64)
    enhanced_stft = np.zeros_like(stft, dtype=np.complex128)

    from scipy.special import exp1
    EULER_GAMMA = 0.57721566490153286
    V_THRESHOLD = 1e-3
    
    def compute_gain(xi_m, gamma_m, X_m, P_noisy_m):
        G_m = np.zeros_like(xi_m)
        valid_mask = (P_noisy_m > 0)
        
        xi_valid = xi_m[valid_mask]
        gamma_valid = gamma_m[valid_mask]
        v_valid = (xi_valid / (1.0 + xi_valid)) * gamma_valid
        G_valid = np.zeros_like(v_valid)
        
        small_mask = (v_valid <= V_THRESHOLD)
        if np.any(small_mask):
            xi_small = xi_valid[small_mask]
            gamma_small = gamma_valid[small_mask]
            v_small = v_valid[small_mask]
            
            G_small = np.zeros_like(xi_small)
            nonzero_xi = (xi_small > 0)
            
            if np.any(nonzero_xi):
                xi_nz = xi_small[nonzero_xi]
                gamma_nz = gamma_small[nonzero_xi]
                v_nz = v_small[nonzero_xi]
                
                term1 = np.sqrt(xi_nz / (1.0 + xi_nz))
                term2 = np.exp(-0.5 * EULER_GAMMA) / np.sqrt(gamma_nz)
                term3 = np.exp(0.5 * v_nz)
                G_small[nonzero_xi] = term1 * term2 * term3
                
            G_valid[small_mask] = G_small
            
        large_mask = ~small_mask
        if np.any(large_mask):
            xi_large = xi_valid[large_mask]
            v_large = v_valid[large_mask]
            
            wiener_factor = xi_large / (1.0 + xi_large)
            exp_integral = exp1(v_large)
            correction = np.exp(0.5 * exp_integral)
            G_valid[large_mask] = wiener_factor * correction
            
        G_m[valid_mask] = G_valid
        return G_m

    for m in range(n_frames):
        # 1. Decision-Directed a-priori SNR (xi)
        if m == 0:
            xi[m] = np.maximum(gamma[m] - 1.0, 0.0)
        else:
            P_enh_prev = np.abs(enhanced_stft[m - 1])**2
            xi[m] = alpha_dd * (P_enh_prev / P_noise_safe[m]) + \
                    (1.0 - alpha_dd) * np.maximum(gamma[m] - 1.0, 0.0)
            
        xi[m] = np.maximum(xi[m], 0.0)
        
        # 2. Base Log-MMSE Gain G_LMMSE(m,k)
        G_lmmse = compute_gain(xi[m], gamma[m], stft[m], P_noisy[m])
                                
        # 3. OM-LSA Geometric Modification
        p = np.clip(spp[m], 0.0, 1.0)
        
        # Log-domain evaluation: G_OMLSA = exp( p*log(G_LMMSE) + (1-p)*log(G_min) )
        G_lmmse_safe = np.maximum(G_lmmse, epsilon)
        log_G_omlsa = p * np.log(G_lmmse_safe) + (1.0 - p) * np.log(G_min)
        G_omlsa = np.exp(log_G_omlsa)

        # Frequency-domain gain smoothing (same 3-bin uniform kernel used by
        # Spectral Subtraction) — suppresses isolated per-bin gain spikes
        # (e.g. from a frame whose window straddles a noise/speech boundary)
        # that would otherwise reconstruct as a narrowband, noise-phase
        # "musical noise" artifact. See apply_wiener_filter_dd for the full note.
        G_omlsa = uniform_filter1d(G_omlsa, size=_FREQ_SMOOTH_BINS, mode="nearest")

        # 4. Reconstruction and X=0 protection
        Y_m = G_omlsa * stft[m]
        zero_x_mask = (P_noisy[m] == 0.0)
        Y_m[zero_x_mask] = 0.0
        
        enhanced_stft[m] = Y_m
        
    return enhanced_stft


def denoise_omlsa(
    signal: np.ndarray,
    sample_rate_hz: int = 44100,
    frame_length: int = 2048,
    hop_length: int | None = None,
    alpha_dd: float = 0.98,
    G_min: float = OMLSA_DEFAULT_GMIN,
    imcra_alpha_s: float = 0.86,
    imcra_alpha_d: float = 0.85,
) -> np.ndarray:
    """Denoise an audio signal using IMCRA Noise Tracking + OM-LSA Gain.

    G_min:
        Minimum gain floor in [0, 1] applied when speech is judged absent.
        Defaults to `OMLSA_DEFAULT_GMIN` (unlike `apply_omlsa_filter`'s own
        default of 0.01) because a slightly higher floor measurably reduces
        an audible artifact at noise/speech transitions ("pre-echo" — see
        `denoise_wiener_dd`'s docstring for the full mechanism) while still
        leaving OM-LSA's SPP-driven suppression well below Spectral
        Subtraction's or Wiener's floor during genuine, sustained silence.
    """
    if hop_length is None:
        hop_length = frame_length // 4

    from dsp_core.spectrum import normalize_samples
    signal_f64 = normalize_samples(signal).astype(np.float64)

    mono = signal_f64.ndim == 1
    channels = [signal_f64] if mono else [signal_f64[:, c] for c in range(signal_f64.shape[1])]
    
    n_samples = len(channels[0])
    enhanced_channels: list[np.ndarray] = []

    for ch_signal in channels:
        # 1. STFT
        S = stft_process(ch_signal, frame_length=frame_length, hop_length=hop_length)
        P_noisy = np.abs(S) ** 2
        
        # 2. IMCRA Noise Estimation & SPP
        initial_frame = get_first_valid_frame_index(frame_length, hop_length)
        P_N_initial = compute_robust_initial_noise(P_noisy, initial_frame)
        
        P_noise, spp = estimate_noise_imcra(
            P_noisy,
            alpha_s=imcra_alpha_s,
            alpha_d=imcra_alpha_d,
            alpha_dd=alpha_dd,
            initial_frame=initial_frame,
            P_N_initial=P_N_initial,
        )
        
        # 3. OM-LSA Enhancement
        S_enh = apply_omlsa_filter(
            S, 
            noise_psd=P_noise, 
            spp=spp, 
            alpha_dd=alpha_dd, 
            G_min=G_min
        )
        
        # 4. ISTFT
        y = istft_process(
            S_enh,
            frame_length=frame_length,
            hop_length=hop_length,
            original_length=n_samples,
        )
        enhanced_channels.append(y)

    result = enhanced_channels[0] if mono else np.stack(enhanced_channels, axis=1)
    return result.astype(np.float32)
