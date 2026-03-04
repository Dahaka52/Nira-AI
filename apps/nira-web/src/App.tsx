import { useState, useEffect, useRef } from 'react'
import { restClient } from './api/rest'
import { wsClient } from './api/ws'
import {
    Activity, MessageSquare, Settings, Zap, Terminal,
    Database, Thermometer, Mic, MicOff,
    Volume2, VolumeX, Play, Sun, Moon, Trash2, Clock,
    Brain, BarChart3
} from 'lucide-react'

interface Msg {
    id: string;
    sender: 'Creator' | 'Nira';
    text: string;
    type?: string;
    tsMs?: number;
    sourceId?: string;
    turnId?: string;
    utteranceId?: string;
    speakerId?: string;
}

const CHAT_CACHE_KEY = 'nira_ui_chat_messages_v1';

const loadCachedMessages = (): Msg[] => {
    if (typeof window === 'undefined') return [];
    try {
        const raw = window.sessionStorage.getItem(CHAT_CACHE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
};

const mergeMessages = (current: Msg[], incoming: Msg[]): Msg[] => {
    if (incoming.length === 0) return current;
    const merged = [...current];
    for (const nextMsg of incoming) {
        const exists = merged.some((cur) => {
            if (cur.sender !== nextMsg.sender) return false;
            if (cur.text !== nextMsg.text) return false;
            if (cur.tsMs !== undefined && nextMsg.tsMs !== undefined) {
                return Math.abs(cur.tsMs - nextMsg.tsMs) <= 2000;
            }
            return true;
        });
        if (!exists) merged.push(nextMsg);
    }
    return merged;
};

function App() {
    const [theme, setTheme] = useState<'dark' | 'light'>('dark');
    const [messages, setMessages] = useState<Msg[]>(() => loadCachedMessages());
    const [input, setInput] = useState('');
    const [fullConfig, setFullConfig] = useState<any>(null);
    const [isThinking, setIsThinking] = useState(false);
    const [status, setStatus] = useState<'idle' | 'listening' | 'thinking' | 'speaking'>('idle');
    const [isMuted, setIsMuted] = useState(false);
    const [isVoiceMuted, setIsVoiceMuted] = useState(false);
    const [activeTab, setActiveTab] = useState<'chat' | 'memory' | 'settings'>('chat');
    const [memoryTab, setMemoryTab] = useState<'read' | 'write' | 'vector'>('read');
    const [history, setHistory] = useState<any[]>([]);
    const [metrics, setMetrics] = useState({
        tps: 0,
        ttftMs: 0,
        e2eTtftMs: 0,
        ttsStartMs: 0,
        e2eTtsStartMs: 0,
        ttsProviderLatencyMs: 0,
        lastRtf: 0
    });

    const scrollRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const processedJobIds = useRef<Set<string>>(new Set());
    const persistTimerRef = useRef<number | null>(null);

    // Theme Switcher
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    useEffect(() => {
        wsClient.connect();
        const unsub = wsClient.subscribe((data) => {
            if (data.event === 'context_conversation_add_text') {
                const payload = data.payload;
                const result = payload?.result;

                // Нам нужен только промежуточный event с контентом
                if (!result || payload?.finished) return;

                const content = String(result.content || '').trim();
                if (!content) return;

                const charName = fullConfig?.prompter?.character_name || 'Нира';
                const sender: Msg['sender'] = result.user === charName ? 'Nira' : 'Creator';

                // В этом блоке интересуют именно реплики пользователя
                if (sender !== 'Creator') return;

                const tsMs = result.timestamp ? Math.round(Number(result.timestamp) * 1000) : Date.now();
                const incomingJobId = String(payload?.job_id || Date.now());

                setMessages(prev => {
                    // Дедуп: typed-сообщение уже добавляется optimistic в handleSend
                    const duplicate = prev.some(m =>
                        m.sender === 'Creator' &&
                        m.text === content &&
                        m.tsMs !== undefined &&
                        Math.abs(m.tsMs - tsMs) <= 1500
                    );
                    if (duplicate) return prev;

                    return [...prev, {
                        id: `ctx-${incomingJobId}-${tsMs}`,
                        sender: 'Creator',
                        text: content,
                        type: 'context_user',
                        tsMs,
                        sourceId: result.source_id,
                        turnId: result.turn_id,
                        utteranceId: result.utterance_id,
                        speakerId: result.speaker_id
                    }];
                });
                return;
            }

            if (data.event === 'stt_status') {
                const result = data.payload?.result || data.payload || {};
                const state = String(result.state || '').toLowerCase();
                if (!state || state === 'partial') return;
                if (state === 'backpressure_merge') return;

                const important = new Set(['timeout', 'unavailable', 'restarting', 'backpressure_drop']);
                if (!important.has(state)) return;

                const source = result.source_id ? ` source=${result.source_id}` : '';
                const reason = result.reason ? ` reason=${result.reason}` : '';
                setMessages(prev => [...prev, {
                    id: `stt-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                    sender: 'Nira',
                    text: `[STT ${state.toUpperCase()}]${source}${reason}`,
                    type: 'stt_status',
                    tsMs: Date.now()
                }]);
                return;
            }

            if (data.event === 'response') {
                if (data.payload?.start) {
                    setStatus('thinking');
                    setIsThinking(true);
                    setMetrics(prev => ({
                        ...prev,
                        tps: 0,
                        ttftMs: 0,
                        e2eTtftMs: 0,
                        ttsStartMs: 0,
                        e2eTtsStartMs: 0,
                        ttsProviderLatencyMs: 0
                    }));
                    return;
                }

                const jobId = data.payload?.job_id;
                const result = data.payload?.result;

                // Глобальный/локальный stop сигнал (barge-in): просто выходим из thinking
                if (result?.event === 'stop_audio') {
                    setIsThinking(false);
                    setStatus('idle');
                    return;
                }

                // Штатная отмена job (не ошибка)
                if (result?.event === 'cancelled') {
                    setIsThinking(false);
                    setStatus('idle');

                    // Показываем системное сообщение только для ручной/внешней отмены,
                    // чтобы не зашумлять чат при обычном voice barge-in.
                    const reason = String(result?.reason || '');
                    const isVoiceInterrupt =
                        reason.includes('user_speaking_significant') ||
                        reason.includes('user_voice_start');

                    if (!isVoiceInterrupt) {
                        setMessages(prev => [...prev, {
                            id: `${jobId}-cancel-${Date.now()}`,
                            sender: 'Nira',
                            text: `[SYSTEM] Request cancelled${reason ? `: ${reason}` : ''}`,
                            tsMs: Date.now()
                        }]);
                    }
                    return;
                }

                if (data.payload?.finished) {
                    setIsThinking(false);
                    setStatus('idle');
                    // Очищаем ID через некоторое время, чтобы не копились
                    setTimeout(() => processedJobIds.current.delete(jobId), 10000);

                    // Возвращаем фокус
                    setTimeout(() => inputRef.current?.focus(), 100);

                    if (data.payload?.success === false) {
                        const errorReason = data.payload?.result?.reason || 'Unknown error';
                        setMessages(prev => [...prev, {
                            id: jobId + '-err',
                            sender: 'Nira',
                            text: `[SYSTEM ERROR] ${errorReason}`,
                            tsMs: Date.now()
                        }]);
                    }
                    return;
                }

                // ФИЛЬТР ДУБЛЕЙ 2.2: Склеиваем чанки одного ответа по Job ID
                if (!result) return;

                const content = result.content || result.filtered_text;

                // Runtime metrics from backend streaming
                if (
                    result.tps !== undefined ||
                    result.ttft_ms !== undefined ||
                    result.e2e_ttft_ms !== undefined ||
                    result.latency !== undefined
                ) {
                    setMetrics(prev => ({
                        ...prev,
                        tps: result.tps ?? prev.tps,
                        ttftMs: result.ttft_ms ?? result.latency ?? prev.ttftMs,
                        e2eTtftMs: result.e2e_ttft_ms ?? prev.e2eTtftMs
                    }));
                }
                if (
                    result.tts_start_ms !== undefined ||
                    result.e2e_tts_start_ms !== undefined ||
                    result.tts_provider_latency_ms !== undefined
                ) {
                    setMetrics(prev => ({
                        ...prev,
                        ttsStartMs: result.tts_start_ms ?? prev.ttsStartMs,
                        e2eTtsStartMs: result.e2e_tts_start_ms ?? prev.e2eTtsStartMs,
                        ttsProviderLatencyMs: result.tts_provider_latency_ms ?? prev.ttsProviderLatencyMs
                    }));
                }
                if (result.tts_rtf !== undefined) {
                    const nextRtf = Number(result.tts_rtf);
                    if (Number.isFinite(nextRtf) && nextRtf > 0) {
                        setMetrics(prev => ({
                            ...prev,
                            lastRtf: nextRtf
                        }));
                    }
                }

                if (content && !result.history) {
                    setMessages(prev => {
                        const existingIdx = prev.findIndex(m => m.id === jobId);
                        if (existingIdx !== -1) {
                            const newMessages = [...prev];
                            // Если это новый чанк, добавляем его к существующему (пробел если нужно)
                            const currentText = newMessages[existingIdx].text;
                            newMessages[existingIdx] = {
                                ...newMessages[existingIdx],
                                text: currentText + content
                            };
                            return newMessages;
                        } else {
                            return [...prev, {
                                id: jobId || Math.random().toString(36),
                                sender: 'Nira',
                                text: content,
                                tsMs: Date.now()
                            }];
                        }
                    });
                }
            }

        });

        loadHistory();
        return () => { unsub(); };
    }, []);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages, isThinking]);

    useEffect(() => {
        if (persistTimerRef.current !== null) {
            window.clearTimeout(persistTimerRef.current);
        }
        persistTimerRef.current = window.setTimeout(() => {
            try {
                // Храним ограниченный хвост, чтобы не раздувать storage.
                const tail = messages.slice(-300);
                window.sessionStorage.setItem(CHAT_CACHE_KEY, JSON.stringify(tail));
            } catch { }
        }, 350);
        return () => {
            if (persistTimerRef.current !== null) {
                window.clearTimeout(persistTimerRef.current);
                persistTimerRef.current = null;
            }
        };
    }, [messages]);

    const loadHistory = async () => {
        try {
            const configRes: any = await restClient.getConfig();
            setFullConfig(configRes.response);
            const charName = configRes.response?.prompter?.character_name || "Нира";

            const res = await restClient.getHistory();
            const fullHistory = res.response || [];
            setHistory(fullHistory);

            const chatMessages: Msg[] = fullHistory
                .filter((h: any) => h.type === 'chat')
                .map((h: any, idx: number) => ({
                    id: `hist-${h.time}-${idx}`,
                    sender: h.user === charName ? 'Nira' : 'Creator',
                    text: h.message,
                    tsMs: h.time ? Math.round(Number(h.time) * 1000) : Date.now()
                }));
            setMessages(prev => mergeMessages(prev, chatMessages));
        } catch (e) { }
    };

    const handleSend = async () => {
        if (!input.trim() || isThinking) return;
        const text = input.trim();
        setInput('');
        setMessages(prev => [...prev, {
            id: 'user-' + Date.now(),
            sender: 'Creator',
            text,
            tsMs: Date.now()
        }]);
        setIsThinking(true);
        setStatus('thinking');
        setTimeout(() => inputRef.current?.focus(), 0);

        try {
            await restClient.sendMessage(text);
        } catch (e) {
            setIsThinking(false);
            setStatus('idle');
            setMessages(prev => [...prev, { id: 'err-' + Date.now(), sender: 'Nira', text: 'LINK ERROR' }]);
        }
    };

    const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');

    return (
        <div className="dashboard">
            {/* TOP BAR */}
            <header className="topbar">
                <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <div className="logo">NIRA</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.7rem', color: 'var(--text-dim)', fontFamily: 'Orbitron' }}>
                        <span className={`indicator ${status !== 'idle' ? 'on' : 'off'}`} />
                        {status.toUpperCase()}
                    </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--border)', padding: '6px 16px', borderRadius: '20px' }}>
                        <Brain size={18} color="var(--accent)" />
                        <span style={{ fontSize: '0.85rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Sassy / Curious</span>
                    </div>

                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button className="send-btn" style={{ padding: '6px 12px', background: isMuted ? '#ef4444' : 'var(--border)' }} onClick={() => setIsMuted(!isMuted)}>
                            {isMuted ? <MicOff size={16} /> : <Mic size={16} />}
                        </button>
                        <button className="send-btn" style={{ padding: '6px 12px', background: isVoiceMuted ? '#f59e0b' : 'var(--border)' }} onClick={() => setIsVoiceMuted(!isVoiceMuted)}>
                            {isVoiceMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                        </button>
                    </div>

                    <button onClick={toggleTheme} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text)' }}>
                        {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
                    </button>

                    <button className="send-btn" style={{ background: 'var(--neon)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Play size={14} fill="currentColor" /> LAUNCH
                    </button>
                </div>
            </header>

            <div className="layout-main">
                {/* LEFT: METRICS & CONTROLS */}
                <aside className="pane">
                    <div className="header"><BarChart3 size={16} /> METRICS</div>
                    <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        <div className="stat-card">
                            <div className="stat-label">Tokens / Sec</div>
                            <div className="stat-value" style={{ color: 'var(--neon)' }}>{metrics.tps || '0.0'} <span style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>live</span></div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">TTFT</div>
                            <div className="stat-value">
                                {(metrics.e2eTtftMs || metrics.ttftMs || 0)}ms
                                <span style={{ fontSize: '0.6rem', color: 'var(--text-dim)', marginLeft: '6px' }}>
                                    {metrics.e2eTtftMs ? 'e2e' : 'llm'}
                                </span>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">TTFAudio</div>
                            <div className="stat-value">
                                {(metrics.e2eTtsStartMs || metrics.ttsStartMs || 0)}ms
                                <span style={{ fontSize: '0.6rem', color: 'var(--text-dim)', marginLeft: '6px' }}>
                                    {metrics.e2eTtsStartMs ? 'e2e' : 'tts'}
                                </span>
                            </div>
                            <div style={{ marginTop: '4px', fontSize: '0.65rem', color: 'var(--text-dim)' }}>
                                provider {(metrics.ttsProviderLatencyMs || 0)}ms
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">RTF (last phrase)</div>
                            <div className="stat-value">
                                {(metrics.lastRtf > 0 ? metrics.lastRtf : 0).toFixed(3)}
                            </div>
                        </div>
                    </div>
                </aside>

                {/* CENTER: CHAT / MEMORY */}
                <main className="pane" style={{ borderRight: 'none' }}>
                    <div className="tabs-header">
                        <button className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>CHAT</button>
                        <button className={`tab-btn ${activeTab === 'memory' ? 'active' : ''}`} onClick={() => setActiveTab('memory')}>MEMORY</button>
                        <button className={`tab-btn ${activeTab === 'settings' ? 'active' : ''}`} onClick={() => setActiveTab('settings')}>LLM_SETTINGS</button>
                    </div>

                    {activeTab === 'chat' ? (
                        <>
                            <div className="chat-container" ref={scrollRef}>
                                {messages.map(m => (
                                    <div key={m.id} className={`msg ${m.sender.toLowerCase()}`}>
                                        <div className="msg-info">{m.sender === 'Creator' ? 'SOURCE' : 'NIRA_AI'}</div>
                                        <div>{m.text}</div>
                                        {m.turnId && (
                                            <div className="msg-info" style={{ opacity: 0.55 }}>
                                                {`${m.sourceId || 'mic'} · turn=${m.turnId}${m.utteranceId ? ` · utt=${m.utteranceId}` : ''}`}
                                            </div>
                                        )}
                                    </div>
                                ))}
                                {isThinking && (
                                    <div className="msg nira" style={{ opacity: 0.6 }}>
                                        <div className="msg-info">CORE</div>
                                        <div className="glow-text">Synthesizing...</div>
                                    </div>
                                )}
                            </div>
                            <div className="input-area">
                                <input ref={inputRef} className="main-input" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSend()} placeholder="Interlink message..." disabled={isThinking} autoFocus />
                                <button className="send-btn" onClick={handleSend} disabled={isThinking}>SEND</button>
                            </div>
                        </>
                    ) : activeTab === 'memory' ? (
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                            <div className="tabs-header" style={{ background: 'transparent' }}>
                                <button className={`tab-btn ${memoryTab === 'read' ? 'active' : ''}`} onClick={() => setMemoryTab('read')}>EVENT_READ</button>
                                <button className={`tab-btn ${memoryTab === 'write' ? 'active' : ''}`} onClick={() => setMemoryTab('write')}>EVENT_WRITE</button>
                                <button className={`tab-btn ${memoryTab === 'vector' ? 'active' : ''}`} onClick={() => setMemoryTab('vector')}>VECTOR_PAD</button>
                            </div>
                            <div style={{ padding: '0', overflowY: 'auto', flex: 1 }}>
                                <table className="memory-table">
                                    <thead>
                                        <tr>
                                            <th>TIME</th>
                                            <th>SOURCE</th>
                                            <th>CONTENT</th>
                                            <th>ACTION</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {history.map((h, i) => (
                                            <tr key={i}>
                                                <td style={{ opacity: 0.5 }}>{new Date(h.time * 1000).toLocaleTimeString()}</td>
                                                <td><span className="status-tag status-active">{h.user || 'SYS'}</span></td>
                                                <td style={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{h.message}</td>
                                                <td><Trash2 size={14} color="#ef4444" style={{ cursor: 'pointer' }} /></td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ) : (
                        <div style={{ flex: 1, padding: '24px', overflowY: 'auto' }}>
                            <div className="header" style={{ background: 'transparent', padding: '0 0 16px 0', borderBottom: '1px solid var(--border)', marginBottom: '20px' }}><Settings size={16} /> LLM PARAMETERS</div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                {[
                                    { key: 'temperature', label: 'Temperature (0.01 - 2.0)', min: 0.01, max: 2.0, step: 0.01 },
                                    { key: 'top_p', label: 'Top P (0.01 - 1.0)', min: 0.01, max: 1.0, step: 0.01 },
                                    { key: 'min_p', label: 'Min P (0.0 - 1.0)', min: 0.0, max: 1.0, step: 0.01 },
                                    { key: 'top_k', label: 'Top K (1 - 100)', min: 1, max: 100, step: 1 }
                                ].map(setting => {
                                    const t2tOp = fullConfig?.operations?.find((o: any) => o.role === 't2t') || {};
                                    const val = t2tOp[setting.key] !== undefined ? t2tOp[setting.key] : setting.max / 2;
                                    return (
                                        <div key={setting.key} className="stat-card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <div className="stat-label" style={{ marginBottom: 0 }}>{setting.label}</div>
                                                <div className="stat-value" style={{ fontSize: '1rem', color: 'var(--neon)' }}>{val}</div>
                                            </div>
                                            <input
                                                type="range" min={setting.min} max={setting.max} step={setting.step} value={val}
                                                onChange={e => {
                                                    const newConfig = { ...fullConfig };
                                                    const ops = newConfig.operations || [];
                                                    const op = ops.find((o: any) => o.role === 't2t');
                                                    if (op) { op[setting.key] = parseFloat(e.target.value); }
                                                    setFullConfig(newConfig);
                                                }}
                                                onMouseUp={async () => {
                                                    try { await restClient.updateConfig(fullConfig); } catch (e) { }
                                                }}
                                                style={{ width: '100%', accentColor: 'var(--accent)' }}
                                            />
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </main>

            </div>

            {/* System HUD intentionally disabled: UI should not pressure realtime pipeline */}
        </div>
    )
}

export default App
