"""
tests/test_denoise_preecho_regression.py — Noise/speech-boundary "pre-echo" fix
=================================================================================

Regression coverage for a "robotic" artifact that could be heard immediately
BEFORE the first real speech in a recording, after an earlier fix had already
resolved a larger, separate cold-start noise-estimation bias.

Root cause (see AGENT_INSTRUCTIONS.md / final report for the full derivation):
STFT frames are analysed with a symmetric Hann window wide enough that a
frame's window support extends on both sides of its nominal centre. A frame
whose window already straddles a noise -> speech boundary legitimately needs
a higher gain once real speech enters it, but that frame's SYNTHESIS window
also spans both sides of the boundary, so overlap-add smears part of its
reconstruction into samples that are, in real time, still before the audible
onset ("pre-echo" -- a structural consequence of applying one gain per
analysis frame). Because that smeared reconstruction is built mostly from
still-noise-derived phase, it is heard as a short, narrowband, incoherent
artifact rather than a natural quiet fade-in.

The fix (frequency-domain gain smoothing, already used by Spectral
Subtraction, extended to Wiener/Log-MMSE/OM-LSA; plus a modest gain floor for
Wiener/Log-MMSE/OM-LSA, which previously had none or too low a floor) reduces
the size of the noise<->speech gain swing without delaying the response to
genuine, sustained speech.

These tests verify the actual mathematical/acoustic behaviour (energy level
and spectral flatness in a short window immediately before a real onset,
compared to a steady noise-only reference window), not merely that a file
is produced.
"""
from __future__ import annotations

import numpy as np
import pytest

from dsp_core.denoise import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    WIENER_DEFAULT_GMIN,
    LOGMMSE_DEFAULT_GMIN,
    OMLSA_DEFAULT_GMIN,
    apply_wiener_filter_dd,
    apply_logmmse_filter,
    denoise_spectral_subtraction,
    denoise_wiener_dd,
    denoise_logmmse,
    denoise_omlsa,
)

SR = 44100
WIN = int(0.010 * SR)  # 10 ms analysis window, ~ one hop at defaults


def _noise_then_speech(noise_dur=2.0, speech_dur=1.0, noise_sigma=0.05, seed=42):
    """G: a long noise-only prefix followed by a speech-like harmonic onset.

    The harmonic burst has a natural 10 ms linear attack ramp (not a
    denoiser-side fade -- this models a real recording, where the onset
    itself is not instantaneous).
    """
    rng = np.random.default_rng(seed)
    n_noise = int(noise_dur * SR)
    n_speech = int(speech_dur * SR)
    noise = rng.normal(0, noise_sigma, n_noise)
    t = np.arange(n_speech) / SR
    f0 = 150.0
    speech = sum(0.15 / h * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 6))
    attack = np.minimum(1.0, t / 0.01)
    speech = speech * attack
    speech_noisy = speech + rng.normal(0, noise_sigma, n_speech)
    return np.concatenate([noise, speech_noisy]), n_noise


def _spectral_flatness(segment):
    spec = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    spec = np.maximum(spec, 1e-12)
    gmean = np.exp(np.mean(np.log(spec)))
    amean = np.mean(spec)
    return gmean / amean  # near 1 = broadband/noise-like; low = tonal/peaky


def _reference_and_pre_onset_windows(y, onset_sample):
    ref_start = onset_sample - int(1.0 * SR)
    ref = y[ref_start: ref_start + WIN]
    pre = y[onset_sample - WIN: onset_sample]
    return ref, pre


class TestNoPreEchoArtifact:
    """The core regression: energy/tonality immediately before a real onset
    must stay close to the steady noise-only reference, for every method."""

    # (name, denoise_fn, kwargs, max_db_above_reference, min_flatness_ratio)
    METHOD_CASES = [
        ("spectral_subtraction", denoise_spectral_subtraction, {}, 3.0, 0.90),
        ("wiener", denoise_wiener_dd, {}, 6.0, 0.85),
        ("logmmse", denoise_logmmse, {}, 6.0, 0.85),
        ("omlsa", denoise_omlsa, {}, 8.0, 0.80),
    ]

    @pytest.mark.parametrize("name,fn,kwargs,max_db,min_flat_ratio", METHOD_CASES)
    def test_pre_onset_energy_close_to_reference(self, name, fn, kwargs, max_db, min_flat_ratio):
        sig, onset = _noise_then_speech()
        y = fn(sig, sample_rate_hz=SR, **kwargs)
        ref, pre = _reference_and_pre_onset_windows(y, onset)

        ref_rms = np.sqrt(np.mean(ref.astype(np.float64) ** 2)) + 1e-12
        pre_rms = np.sqrt(np.mean(pre.astype(np.float64) ** 2)) + 1e-12
        excess_db = 20.0 * np.log10(pre_rms / ref_rms)

        assert excess_db < max_db, (
            f"{name}: pre-onset window is {excess_db:.2f} dB above the steady "
            f"noise-only reference (limit {max_db} dB) -- indicates a residual "
            f"pre-echo energy spike immediately before speech."
        )

    @pytest.mark.parametrize("name,fn,kwargs,max_db,min_flat_ratio", METHOD_CASES)
    def test_pre_onset_not_tonal_relative_to_reference(self, name, fn, kwargs, max_db, min_flat_ratio):
        sig, onset = _noise_then_speech()
        y = fn(sig, sample_rate_hz=SR, **kwargs)
        ref, pre = _reference_and_pre_onset_windows(y, onset)

        ref_flat = _spectral_flatness(ref)
        pre_flat = _spectral_flatness(pre)

        # A "robotic"/tonal pre-echo artifact concentrates energy in a few
        # bins, collapsing spectral flatness well below the steady noise
        # reference. Require the pre-onset window to stay broadband-ish,
        # i.e. not collapse to less than min_flat_ratio of the reference's
        # own flatness.
        assert pre_flat >= min_flat_ratio * ref_flat, (
            f"{name}: pre-onset spectral flatness {pre_flat:.3f} collapsed well "
            f"below the reference noise window's {ref_flat:.3f} "
            f"(ratio {pre_flat / ref_flat:.3f} < {min_flat_ratio}) -- indicates "
            f"tonal/narrowband pre-echo energy immediately before speech."
        )

    @pytest.mark.parametrize("name,fn,kwargs,max_db,min_flat_ratio", METHOD_CASES)
    def test_genuine_onset_response_not_delayed(self, name, fn, kwargs, max_db, min_flat_ratio):
        """The fix must not come at the cost of a sluggish response to real,
        sustained speech: gain at and shortly after the true onset should
        still recover strongly, not stay suppressed for many extra frames."""
        sig, onset = _noise_then_speech()
        y = fn(sig, sample_rate_hz=SR, **kwargs)

        on_rms = np.sqrt(np.mean(y[onset + 2 * WIN: onset + 3 * WIN].astype(np.float64) ** 2))
        in_rms = np.sqrt(np.mean(sig[onset + 2 * WIN: onset + 3 * WIN].astype(np.float64) ** 2))
        recovery_db = 20.0 * np.log10((on_rms + 1e-12) / (in_rms + 1e-12))

        # 20-30ms into real, sustained speech, output should be within a few
        # dB of the input (i.e. gain has opened up), not still heavily
        # suppressed as if the algorithm were still treating it as noise.
        # (Spectral Subtraction's own pre-existing temporal gain smoothing,
        # unrelated to this fix, makes it settle slightly slower than the
        # other three methods, hence the wider margin here.)
        assert recovery_db > -4.0, (
            f"{name}: output is still {recovery_db:.2f} dB below input 20-30ms "
            f"into sustained speech -- the fix may have over-suppressed "
            f"genuine onset response."
        )


class TestGainFloorsApplied:
    """Direct, low-level checks that the new gain floors actually take
    effect (and that the low-level primitives keep their old, un-floored
    default, preserving existing behavioural guarantees)."""

    def test_wiener_pipeline_default_gmin_matches_constant(self):
        sig, _ = _noise_then_speech(noise_dur=0.5, speech_dur=0.2)
        y_default = denoise_wiener_dd(sig, sample_rate_hz=SR)
        y_explicit = denoise_wiener_dd(sig, sample_rate_hz=SR, g_min=WIENER_DEFAULT_GMIN)
        np.testing.assert_allclose(y_default, y_explicit)

    def test_logmmse_pipeline_default_gmin_matches_constant(self):
        sig, _ = _noise_then_speech(noise_dur=0.5, speech_dur=0.2)
        y_default = denoise_logmmse(sig, sample_rate_hz=SR)
        y_explicit = denoise_logmmse(sig, sample_rate_hz=SR, g_min=LOGMMSE_DEFAULT_GMIN)
        np.testing.assert_allclose(y_default, y_explicit)

    def test_omlsa_pipeline_default_gmin_matches_constant(self):
        sig, _ = _noise_then_speech(noise_dur=0.5, speech_dur=0.2)
        y_default = denoise_omlsa(sig, sample_rate_hz=SR)
        y_explicit = denoise_omlsa(sig, sample_rate_hz=SR, G_min=OMLSA_DEFAULT_GMIN)
        np.testing.assert_allclose(y_default, y_explicit)

    def test_apply_wiener_filter_dd_default_is_unfloored(self):
        """Low-level primitive must keep g_min=0.0 by default so existing
        exact-zero-gain unit tests (e.g. the DD-recursion correctness test)
        remain valid; the floor is opt-in at this level."""
        n_frames, n_freq = 2, 5
        stft = np.ones((n_frames, n_freq), dtype=np.complex128)
        noise_psd = np.ones((n_frames, n_freq), dtype=np.float64) * 2.0
        enhanced = apply_wiener_filter_dd(stft, noise_psd, alpha_dd=0.98)
        # With gamma=0.5 at both frames, xi=0 throughout -> gain must reach
        # exactly 0 without an explicit floor.
        assert np.all(np.abs(enhanced) < 1e-10)

    def test_apply_wiener_filter_dd_gmin_raises_floor(self):
        n_frames, n_freq = 2, 5
        stft = np.ones((n_frames, n_freq), dtype=np.complex128)
        noise_psd = np.ones((n_frames, n_freq), dtype=np.float64) * 2.0
        enhanced = apply_wiener_filter_dd(stft, noise_psd, alpha_dd=0.98, g_min=0.1)
        gain = np.abs(enhanced) / (np.abs(stft) + 1e-20)
        assert np.all(gain >= 0.1 - 1e-9)

    def test_apply_logmmse_filter_gmin_raises_floor(self):
        n_frames, n_freq = 10, 129
        rng = np.random.default_rng(7)
        stft = rng.normal(0, 0.01, size=(n_frames, n_freq)) + 1j * rng.normal(0, 0.01, size=(n_frames, n_freq))
        noise_psd = rng.uniform(0.5, 2.0, size=(n_frames, n_freq))
        enhanced = apply_logmmse_filter(stft, noise_psd, alpha_dd=0.98, g_min=0.1)
        gain = np.abs(enhanced) / (np.abs(stft) + 1e-20)
        assert np.all(gain >= 0.1 - 1e-6)


class TestFullPipelineSanity:
    """General mathematical-behaviour checks requested alongside the fix:
    finiteness, boundedness, and that ordinary startup/silence behaviour
    (already covered elsewhere) still holds through the new code paths."""

    METHODS = [denoise_spectral_subtraction, denoise_wiener_dd, denoise_logmmse, denoise_omlsa]

    @pytest.mark.parametrize("fn", METHODS)
    def test_startup_silence_is_finite_and_quiet(self, fn):
        signal = np.zeros(SR, dtype=np.float64)
        y = fn(signal, sample_rate_hz=SR)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) < 1e-6

    @pytest.mark.parametrize("fn", METHODS)
    def test_noise_only_prefix_then_speech_is_finite_and_bounded(self, fn):
        sig, _ = _noise_then_speech()
        y = fn(sig, sample_rate_hz=SR)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) <= 1.5  # generous bound; output should not blow up

    @pytest.mark.parametrize("fn", METHODS)
    def test_output_length_matches_input(self, fn):
        sig, _ = _noise_then_speech(noise_dur=0.3, speech_dur=0.2)
        y = fn(sig, sample_rate_hz=SR)
        assert len(y) == len(sig)
