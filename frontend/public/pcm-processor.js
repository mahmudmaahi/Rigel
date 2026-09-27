// pcm-processor.js
// AudioWorkletProcessor to extract raw Float32 PCM from the microphone

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // We want to send chunks of e.g. 2048 or 4096 samples to avoid
    // overwhelming the WebSocket with tiny 128-sample frames.
    this.bufferSize = 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.offset = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    // Use the first channel (mono)
    const channelData = input[0];

    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.offset] = channelData[i];
      this.offset++;

      if (this.offset >= this.bufferSize) {
        // Send a copy of the buffer to the main thread
        this.port.postMessage(this.buffer.slice());
        this.offset = 0;
      }
    }

    return true; // Keep the processor alive
  }
}

registerProcessor("pcm-processor", PCMProcessor);
