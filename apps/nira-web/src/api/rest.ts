export interface ApiResponse<T> {
    status: number;
    message: string;
    response: T;
}

export interface PipelineStats {
    current_job_id: string | null;
    queue_size: number;
    status: 'active' | 'idle';
    loaded_operations: Record<string, any>;
    stt?: Record<string, any>;
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

    async sendMessage(content: string) {
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
            body: JSON.stringify({ include_audio: false }),
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
}

export const restClient = new RestClient();
