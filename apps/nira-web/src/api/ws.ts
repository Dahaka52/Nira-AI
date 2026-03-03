export type WsHandler = (data: any) => void;

class WsClient {
    private socket: WebSocket | null = null;
    private handlers: Set<WsHandler> = new Set();

    private getUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname === 'localhost' ? 'localhost:7272' : '127.0.0.1:7272';
        return `${protocol}//${host}/`;
    }

    connect() {
        // Если сокет уже в процессе открытия или открыт — ничего не делаем
        if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
            console.log('[WS] Already connected or connecting...');
            return;
        }

        const url = this.getUrl();
        console.log('[WS] Connecting to:', url);

        try {
            this.socket = new WebSocket(url);

            this.socket.onopen = () => {
                console.log('[WS] Connection established');
            };

            this.socket.onmessage = (event) => {
                try {
                    const rawData = JSON.parse(event.data);
                    const data = Array.isArray(rawData) ? rawData[0] : rawData;

                    const formatted = {
                        event: data.message || data.event,
                        payload: data.response || data.payload || data,
                        status: data.status
                    };

                    this.handlers.forEach(h => h(formatted));
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };

            this.socket.onclose = (event) => {
                // Если это не намеренное закрытие, пробуем переподключиться
                if (event.code !== 1000) {
                    console.log(`[WS] Connection closed (${event.code}). Reconnecting in 3s...`);
                    setTimeout(() => this.connect(), 3000);
                }
            };

            this.socket.onerror = (error) => {
                console.error('[WS] WebSocket Error:', error);
            };
        } catch (e) {
            console.error('[WS] Setup error:', e);
        }
    }

    disconnect() {
        if (this.socket) {
            console.log('[WS] Intentional disconnect');
            this.socket.close(1000, 'Intentional disconnect');
            this.socket = null;
        }
    }

    subscribe(h: WsHandler) {
        this.handlers.add(h);
        return () => this.handlers.delete(h);
    }
}

export const wsClient = new WsClient();
