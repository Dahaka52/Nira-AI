export interface ApiResponse<T> {
    status: number;
    message: string;
    response: T;
}

export interface PipelineTelemetry {
    job_id: string;
    input_mode?: string;
    timestamp?: number;
    total_latency_ms?: number;
    response_latency_ms?: number;
    user_speech_duration_ms?: number;
    turn_taking_latency_ms?: number;
    start_point?: string;
    end_point?: string;
    stt?: {
        provider?: string;
        model?: string;
        type?: string;
        latency_ms?: number;
        confidence?: number;
        [key: string]: any;
    };
    queue_overhead_ms?: number;
    llm?: {
        provider?: string;
        model?: string;
        type?: string;
        ttft_ms?: number;
        e2e_ttft_ms?: number;
        duration_ms?: number;
        token_count?: number;
        char_count?: number;
        tps?: number;
        first_sentence_ms?: number;
        [key: string]: any;
    };
    tts?: {
        provider?: string;
        model?: string;
        type?: string;
        ttfa_ms?: number;
        first_audio_ms?: number;
        e2e_voice_start_ms?: number;
        [key: string]: any;
    };
    total_pipeline_ms?: number;
    [key: string]: any;
}

export interface DiscordBridgeTelemetry {
    online?: boolean;
    connected_to_voice?: boolean;
    channel_name?: string | null;
    guild_id?: string | number | null;
    channel_id?: string | number | null;
    is_playing?: boolean;
    voice_ping_ms?: number | null;
    gateway_ping_ms?: number | null;
    members?: Array<{ id: string; name: string }>;
    updated_at?: number;
    [key: string]: any;
}

export interface ActiveProviders {
    stt?: {
        id?: string;
        name?: string;
        type?: string;
        model?: string;
        [key: string]: any;
    };
    t2t?: {
        id?: string;
        name?: string;
        type?: string;
        model?: string;
        [key: string]: any;
    };
    tts?: {
        id?: string;
        name?: string;
        type?: string;
        model?: string;
        [key: string]: any;
    };
    [key: string]: any;
}

export interface SystemStats {
    cpu?: number;
    ram?: number;
    gpus?: Array<{
        id: number;
        name: string;
        load: number;
        mem_used: number;
        mem_total: number;
        temp: number;
        [key: string]: any;
    }>;
    [key: string]: any;
}

export interface PipelineStats {
    current_job_id: string | null;
    queue_size: number;
    status: 'active' | 'idle';
    loaded_operations: Record<string, any>;
    stt?: Record<string, any>;
    active_providers?: ActiveProviders;
    discord_bridge?: DiscordBridgeTelemetry;
    system?: SystemStats;
    audio_output_mode?: 'discord' | 'local';
    telemetry?: PipelineTelemetry | null;
    telemetry_history?: PipelineTelemetry[];
}

class RestClient {
    // Используем относительный путь, чтобы сработал Vite Proxy (см. vite.config.ts)
    private baseUrl = '/api';

    private async request<T>(path: string, options?: RequestInit): Promise<ApiResponse<T>> {
        try {
            const response = await fetch(`${this.baseUrl}${path}`, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options?.headers,
                },
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (e) {
            console.error(`[REST] Error calling ${path}:`, e);
            throw e;
        }
    }

    async getPipeline(): Promise<ApiResponse<PipelineStats>> {
        return this.request<PipelineStats>('/pipeline');
    }

    async getConfig() {
        return this.request('/config');
    }

    async sendMessage(content: string, includeAudio: boolean = true) {
        // 1. Добавляем текст в контекст
        await this.request('/context/conversation/text', {
            method: 'POST',
            body: JSON.stringify({
                user: 'Creator',
                content,
                timestamp: Math.floor(Date.now() / 1000)
            }),
        });
        // 2. Запускаем генерацию ответа
        return this.request('/response', {
            method: 'POST',
            body: JSON.stringify({ include_audio: includeAudio }),
        });
    }

    async updateConfig(config: any) {
        return this.request('/config/update', {
            method: 'PUT',
            body: JSON.stringify(config),
        });
    }

    async getHistory() {
        return this.request<any[]>('/context/history');
    }

    async setOutputMode(mode: 'discord' | 'local'): Promise<ApiResponse<{ ok: boolean; mode: string }>> {
        return this.request('/output/mode', {
            method: 'POST',
            body: JSON.stringify({ mode }),
        });
    }

    async clearTelemetry(): Promise<ApiResponse<{ ok: boolean }>> {
        return this.request('/pipeline/telemetry', {
            method: 'DELETE',
        });
    }
}

export const restClient = new RestClient();
