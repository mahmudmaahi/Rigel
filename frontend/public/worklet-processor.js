class VadProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    // Check if we have audio input
    if (input && input.length > 0) {
      // Deterministic Mono Policy: Strictly use inputs[0][0]
      const channelData = input[0];
      
      if (channelData && channelData.length > 0) {
        // Copy the data so it doesn't get overwritten while posting
        const buffer = new Float32Array(channelData);
        // Transfer the buffer to avoid cloning overhead
        this.port.postMessage(buffer, [buffer.buffer]);
      }
    }

    return true; // Keep the processor alive
  }
}

registerProcessor('vad-processor', VadProcessor);
