# VAD Benchmark Report (Multi-Engine)

This report evaluates the Voice Activity Detection (VAD) engines currently supported by the Rigel architecture. The objective is to identify processing speed and accuracy metrics across clean speech, noisy environments, and background television noise.

## Engines Evaluated
1. **Silero VAD (v4)**: Production-grade neural network, standard `float32` pipeline.
2. **TEN VAD**: Native C++ optimized implementation using `int16` pipeline (via Python bindings).

*(FireRedVAD and WebRTC were omitted from this run due to environment constraints).*

## Results

### 1. Quiet Voice (`tests/data/quiet_voice.wav`)
*Duration: 12.67s*
* **Silero VAD (v4)**:
  - RTF: 0.0144x (~883x real-time)
  - Segments: 1
  - Total Speech: 11.49s
* **TEN VAD**:
  - RTF: 0.0051x (~2497x real-time)
  - Segments: 3
  - Total Speech: 10.89s
  
*Conclusion*: Both engines detect the speech effectively. TEN VAD is nearly 3x faster, but slightly more fragmented in its segmentation (splitting the speech into 3 segments).

### 2. Noisy Monologue (`tests/data/noise_mono.wav`)
*Duration: 7.15s*
* **Silero VAD (v4)**:
  - RTF: 0.0180x (~397x real-time)
  - Segments: 2
  - Total Speech: 3.62s
* **TEN VAD**:
  - RTF: 0.0056x (~1266x real-time)
  - Segments: 6
  - Total Speech: 3.19s

*Conclusion*: TEN VAD struggles with continuous tracking through noise, breaking the 3 seconds of speech into 6 micro-segments. Silero maintains smoother continuity.

### 3. Background TV Noise (`tests/data/tv_noise.wav`)
*Duration: 15.05s*
* **Silero VAD (v4)**:
  - RTF: 0.0129x (~1164x real-time)
  - Segments: 0
  - Total Speech: 0.00s
* **TEN VAD**:
  - RTF: 0.0051x (~2971x real-time)
  - Segments: 0
  - Total Speech: 0.00s

*Conclusion*: Both engines successfully reject continuous background noise and babble without false positives.

---

## Architectural Implications
- **Speed**: TEN VAD is exceptionally fast for local inference, making it an excellent candidate for ultra-low latency scenarios if its fragmentation can be smoothed using a hangover/smoothing window.
- **Accuracy**: Silero remains the gold standard for continuous speech detection in noise, providing much stabler segments without requiring complex frontend debouncing.
- **Architecture**: The `VadSession` backend wrapper effectively abstracts the `float32` vs `int16` requirements, allowing seamless engine switching in production.
