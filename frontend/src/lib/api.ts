const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// Status types (Module 01)
// ---------------------------------------------------------------------------

export type ProjectStatus = {
  project: string;
  team: string;
  phase: string;
  module: string;
  api: string;
  dsp_core: string;
  features: string[];
};

// ---------------------------------------------------------------------------
// Audio metadata types (Module 02)
// ---------------------------------------------------------------------------

export type AudioMetadata = {
  filename: string;
  content_type: string;
  format: string;
  sample_rate_hz: number;
  channels: number;
  samples_per_channel: number;
  total_samples: number;
  duration_seconds: number;
  bit_depth: number | null;
  sample_summary: {
    array_shape: number[];
    dtype: string;
    min_value: number;
    max_value: number;
    mean_value: number;
    representation: string;
  };
};

export type AudioUploadResponse = {
  status: "loaded";
  metadata: AudioMetadata;
};

// ---------------------------------------------------------------------------
// Waveform types (Module 03)
//
// Timestamp convention: time_seconds in each WaveformBin is the **start time**
// of the bin chunk — i.e. the time of the first sample in that bin.
// This is used consistently for rendering tick marks and for seeking when the
// user clicks on the waveform.
// ---------------------------------------------------------------------------

export type WaveformBin = {
  /** Start time of the bin in seconds. */
  time_seconds: number;
  min_amplitude: number;
  max_amplitude: number;
};

export type ChannelWaveform = {
  channel_index: number;
  bins: WaveformBin[];
};

export type WaveformData = {
  n_channels: number;
  n_bins: number;
  duration_seconds: number;
  sample_rate_hz: number;
  channels: ChannelWaveform[];
};

export type AudioAnalyzeResponse = {
  status: "analyzed";
  metadata: AudioMetadata;
  waveform: WaveformData;
};

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function getProjectStatus(): Promise<ProjectStatus> {
  const response = await fetch(`${API_BASE_URL}/api/status`, {
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Status request failed with ${response.status}`);
  }

  return response.json() as Promise<ProjectStatus>;
}

export async function uploadAudio(file: File): Promise<AudioUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/audio/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "Audio upload failed.";
    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (errorBody.detail) {
        message = errorBody.detail;
      }
    } catch {
      message = `Audio upload failed with ${response.status}.`;
    }

    throw new Error(message);
  }

  return response.json() as Promise<AudioUploadResponse>;
}

/**
 * Upload an audio file and receive both metadata and decimated waveform data.
 * This is the Module 03 endpoint; it replaces the Module 02 upload call in the
 * primary product flow.
 */
export async function analyzeAudio(file: File): Promise<AudioAnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/audio/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "Audio analysis failed.";
    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (errorBody.detail) {
        message = errorBody.detail;
      }
    } catch {
      message = `Audio analysis failed with ${response.status}.`;
    }

    throw new Error(message);
  }

  return response.json() as Promise<AudioAnalyzeResponse>;
}

// ---------------------------------------------------------------------------
// Spectrum types (Module 04)
//
// magnitude_dbfs: decibels relative to digital full scale.
//   0 dBFS = full-scale amplitude (normalized signal value of 1.0).
//   All real-world signals are below 0 dBFS.
//   This is NOT acoustic sound-pressure dB.
// ---------------------------------------------------------------------------

export type SpectrumBin = {
  /** Centre frequency of this display bin in Hz. */
  frequency_hz: number;
  /**
   * Peak one-sided amplitude magnitude in dBFS within this display bin.
   * 0 dBFS = full-scale. Typical values: -80 dBFS (quiet) to 0 dBFS (loud).
   */
  magnitude_dbfs: number;
};

export type ChannelSpectrum = {
  channel_index: number;
  bins: SpectrumBin[];
};

export type SpectrumData = {
  n_channels: number;
  /** Full FFT length used (N = samples per channel). */
  n_fft_full: number;
  /** Number of display bins returned (≤ 1024). */
  n_display_bins: number;
  /** Δf = Fs / N — raw FFT frequency resolution in Hz. NOT display bin width. */
  frequency_resolution_hz: number;
  /** Theoretical Nyquist frequency = Fs / 2. */
  nyquist_hz: number;
  duration_seconds: number;
  sample_rate_hz: number;
  channels: ChannelSpectrum[];
};

export type AudioSpectrumResponse = {
  status: "spectrum_computed";
  metadata: AudioMetadata;
  spectrum: SpectrumData;
};

/**
 * Upload an audio file and receive its one-sided magnitude spectrum.
 * This is the Module 04 endpoint. Returns up to 1024 display bins per channel
 * with peak-preserving reduction from the full FFT result.
 */
export async function fetchSpectrum(file: File): Promise<AudioSpectrumResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/audio/spectrum`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "Spectrum computation failed.";
    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (errorBody.detail) {
        message = errorBody.detail;
      }
    } catch {
      message = `Spectrum computation failed with ${response.status}.`;
    }

    throw new Error(message);
  }

  return response.json() as Promise<AudioSpectrumResponse>;
}

// ---------------------------------------------------------------------------
// Spectrogram types (Module 05)
//
// The spectrogram is a 2-D time-frequency magnitude representation produced by
// the Short-Time Fourier Transform (STFT).
//
// magnitudes[t][f]:
//   dBFS value at display time column t and frequency row f.
//   0 dBFS = full-scale amplitude; typical values are negative.
//   Frequency rows are sorted ascending (index 0 = near-DC, last = near-Nyquist).
// ---------------------------------------------------------------------------

export type SpectrogramChannel = {
  channel_index: number;
  /** Centre time in seconds for each display column. */
  time_frames: number[];
  /** Centre frequency in Hz for each display row (ascending). */
  freq_bins: number[];
  /** 2-D dBFS magnitude matrix: magnitudes[t][f]. */
  magnitudes: number[][];
};

export type SpectrogramData = {
  n_channels: number;
  /** STFT window size in samples (L). */
  frame_length: number;
  /** STFT hop size in samples (H). */
  hop_length: number;
  /** Total STFT frames before display downsampling. */
  n_frames_full: number;
  /** Total one-sided frequency bins per frame (L//2 + 1). */
  n_freq_bins_full: number;
  /** Number of time columns in the returned display matrix. */
  n_time_display: number;
  /** Number of frequency rows in the returned display matrix. */
  n_freq_display: number;
  /** Δf = Fs / L — raw STFT frequency resolution in Hz. */
  frequency_resolution_hz: number;
  /** Theoretical Nyquist frequency = Fs / 2. */
  nyquist_hz: number;
  /** Δt = H / Fs — time spacing between raw STFT frames. */
  time_resolution_seconds: number;
  duration_seconds: number;
  sample_rate_hz: number;
  /** Analysis window name (e.g. "hann"). */
  window: string;
  channels: SpectrogramChannel[];
};

export type AudioSpectrogramResponse = {
  status: "spectrogram_computed";
  metadata: AudioMetadata;
  spectrogram: SpectrogramData;
};

/**
 * Upload an audio file and receive its STFT spectrogram.
 * This is the Module 05 endpoint.
 *
 * Returns a 2-D [n_time_display × n_freq_display] dBFS magnitude matrix
 * per channel, downsampled to at most 512 × 256 for display.
 */
export async function fetchSpectrogram(file: File, window: string = "hann"): Promise<AudioSpectrogramResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("window", window);

  const response = await fetch(`${API_BASE_URL}/api/audio/spectrogram`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "Spectrogram computation failed.";
    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (errorBody.detail) {
        message = errorBody.detail;
      }
    } catch {
      message = `Spectrogram computation failed with ${response.status}.`;
    }

    throw new Error(message);
  }

  return response.json() as Promise<AudioSpectrogramResponse>;
}

// ---------------------------------------------------------------------------
// Filtering types and API (Module 06)
// ---------------------------------------------------------------------------

export type FilterType = "lowpass" | "highpass" | "bandpass" | "bandstop" | "peaking";
export type FilterFamily = "butterworth" | "chebyshev1" | "chebyshev2" | "elliptic" | "bessel" | "fir_window" | "fir_remez";

export type FilterResponseData = {
  filter_type: FilterType;
  family: FilterFamily;
  order: number;
  sample_rate_hz: number;
  cutoff_hz: number | null;
  low_hz: number | null;
  high_hz: number | null;
  center_hz: number | null;
  frequencies_hz: number[];
  magnitude_db: number[];
};

export type FilterParams = {
  filterType: FilterType;
  family?: FilterFamily;
  order: number;
  cutoffHz?: number;
  lowHz?: number;
  highHz?: number;
  rippleDb?: number;
  attenuationDb?: number;
  firWindow?: string;
  transitionBandwidthHz?: number;
  centerHz?: number;
  gainDb?: number;
  qFactor?: number;
};

/**
 * Fetch the theoretical frequency response of a digital filter.
 * No audio file is required — only sample rate + filter parameters.
 */
export async function fetchFilterResponse(
  sampleRateHz: number,
  params: FilterParams,
): Promise<FilterResponseData> {
  const body = {
    filter_type: params.filterType,
    family: params.family ?? "butterworth",
    order: params.order,
    sample_rate_hz: sampleRateHz,
    cutoff_hz: params.cutoffHz ?? null,
    low_hz: params.lowHz ?? null,
    high_hz: params.highHz ?? null,
    ripple_db: params.rippleDb ?? 1.0,
    attenuation_db: params.attenuationDb ?? 40.0,
    fir_window: params.firWindow ?? "hamming",
    transition_bandwidth_hz: params.transitionBandwidthHz ?? 200.0,
    center_hz: params.centerHz ?? null,
    gain_db: params.gainDb ?? 0.0,
    q_factor: params.qFactor ?? 1.0,
  };

  const response = await fetch(`${API_BASE_URL}/api/audio/filter/response`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let message = "Failed to compute filter frequency response.";
    try {
      const err = (await response.json()) as { detail?: string };
      if (err.detail) message = err.detail;
    } catch { /* ignore */ }
    throw new Error(message);
  }

  return response.json() as Promise<FilterResponseData>;
}

/**
 * Apply a digital filter to an audio file.
 * Returns the filtered audio as a Blob (WAV).
 */
export async function applyFilter(
  file: File,
  params: FilterParams,
): Promise<Blob> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("filter_type", params.filterType);
  formData.append("family", params.family ?? "butterworth");
  formData.append("order", String(params.order));
  if (params.cutoffHz !== undefined) formData.append("cutoff_hz", String(params.cutoffHz));
  if (params.lowHz !== undefined) formData.append("low_hz", String(params.lowHz));
  if (params.highHz !== undefined) formData.append("high_hz", String(params.highHz));
  if (params.rippleDb !== undefined) formData.append("ripple_db", String(params.rippleDb));
  if (params.attenuationDb !== undefined) formData.append("attenuation_db", String(params.attenuationDb));
  if (params.firWindow !== undefined) formData.append("fir_window", params.firWindow);
  if (params.transitionBandwidthHz !== undefined) formData.append("transition_bandwidth_hz", String(params.transitionBandwidthHz));
  if (params.centerHz !== undefined) formData.append("center_hz", String(params.centerHz));
  if (params.gainDb !== undefined) formData.append("gain_db", String(params.gainDb));
  if (params.qFactor !== undefined) formData.append("q_factor", String(params.qFactor));

  const response = await fetch(`${API_BASE_URL}/api/audio/filter`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "Audio filtering failed.";
    try {
      const err = (await response.json()) as { detail?: string };
      if (err.detail) message = err.detail;
    } catch { /* ignore */ }
    throw new Error(message);
  }

  return response.blob();
}

// ---------------------------------------------------------------------------
// Denoising types (Module 07)
// ---------------------------------------------------------------------------

export type DenoiseMethod = "spectral_subtraction" | "wiener" | "logmmse" | "imcra";

export type DenoiseParams = {
  method: DenoiseMethod;
  alpha?: number;
  beta?: number;
  alphaDd?: number;
  noiseAlphaS?: number;
  noiseBias?: number;
  gMin?: number;
  imcraAlphaS?: number;
  imcraAlphaD?: number;
};

export async function applyDenoise(
  file: File,
  params: DenoiseParams,
): Promise<Blob> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("method", params.method);

  if (params.alpha !== undefined)
    formData.append("alpha", params.alpha.toString());
  if (params.beta !== undefined)
    formData.append("beta", params.beta.toString());
  if (params.alphaDd !== undefined)
    formData.append("alpha_dd", params.alphaDd.toString());
  if (params.noiseAlphaS !== undefined)
    formData.append("noise_alpha_s", params.noiseAlphaS.toString());
  if (params.noiseBias !== undefined)
    formData.append("noise_bias", params.noiseBias.toString());
  if (params.gMin !== undefined)
    formData.append("g_min", params.gMin.toString());
  if (params.imcraAlphaS !== undefined)
    formData.append("imcra_alpha_s", params.imcraAlphaS.toString());
  if (params.imcraAlphaD !== undefined)
    formData.append("imcra_alpha_d", params.imcraAlphaD.toString());

  const response = await fetch(`${API_BASE_URL}/api/audio/denoise`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "Denoising failed.";
    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (errorBody.detail) {
        message = errorBody.detail;
      }
    } catch {
      message = `Denoising failed with ${response.status}.`;
    }
    throw new Error(message);
  }

  return await response.blob();
}

