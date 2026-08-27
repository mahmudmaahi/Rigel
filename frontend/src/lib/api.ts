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
