export class PCMPlayer {
    private audioCtx: AudioContext | null = null;
    private nextTime: number = 0;
    private sampleRate: number;
    private isMuted: boolean = false;

    constructor(sampleRate: number = 44100) {
        this.sampleRate = sampleRate;
    }

    public init() {
        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({
                sampleRate: this.sampleRate,
            });
            this.nextTime = this.audioCtx.currentTime + 0.1; // небольшой буфер
        }
    }

    public setMuted(muted: boolean) {
        this.isMuted = muted;
    }

    public stop() {
        if (this.audioCtx) {
            this.audioCtx.close();
            this.audioCtx = null;
        }
        this.nextTime = 0;
    }

    // Принимает base64 строку с 16-bit PCM (mono)
    public feedBase64(base64Data: string, sr: number = 44100) {
        if (this.isMuted) return;
        
        if (!this.audioCtx) {
            this.init();
        }

        // Если контекст был приостановлен (например, из-за автоплея браузера)
        if (this.audioCtx?.state === 'suspended') {
            this.audioCtx.resume();
        }

        if (this.sampleRate !== sr && this.audioCtx) {
            // Если частота сменилась (редко бывает, но всё же)
            this.sampleRate = sr;
            this.stop();
            this.init();
        }

        const binaryString = window.atob(base64Data);
        const len = binaryString.length;
        
        // 16-bit PCM = 2 bytes per sample
        const numSamples = len / 2;
        const floatArray = new Float32Array(numSamples);

        const dataView = new DataView(new ArrayBuffer(len));
        for (let i = 0; i < len; i++) {
            dataView.setUint8(i, binaryString.charCodeAt(i));
        }

        // Конвертация int16 (little endian) в float32
        for (let i = 0; i < numSamples; i++) {
            const int16 = dataView.getInt16(i * 2, true); 
            floatArray[i] = int16 / 32768.0; 
        }

        this.playBuffer(floatArray);
    }

    private playBuffer(floatArray: Float32Array) {
        if (!this.audioCtx) return;

        const audioBuffer = this.audioCtx.createBuffer(1, floatArray.length, this.sampleRate);
        audioBuffer.getChannelData(0).set(floatArray);

        const source = this.audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioCtx.destination);

        // Синхронизация времени для гладкого воспроизведения чанков
        const currentTime = this.audioCtx.currentTime;
        if (this.nextTime < currentTime) {
            this.nextTime = currentTime + 0.1; 
        }

        source.start(this.nextTime);
        this.nextTime += audioBuffer.duration;
    }
}
