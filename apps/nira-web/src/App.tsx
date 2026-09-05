import { useState, useEffect, useRef } from 'react'
import {
    restClient,
    PipelineTelemetry,
    DiscordBridgeTelemetry,
    ActiveProviders,
    SystemStats
} from './api/rest'
import { wsClient } from './api/ws'
import {
    Activity, MessageSquare, Settings, Zap, Terminal,
    Database, Thermometer, Mic, MicOff,
    Volume2, VolumeX, Play, Sun, Moon, Trash2, Clock,
    Brain, BarChart3, Radio, Server, Cpu, Layers, CheckCircle2, Gauge, Headphones
} from 'lucide-react'
import { PCMPlayer } from './api/pcmPlayer'

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
    
    // Telemetry and Provider States
    const [pipelineTelemetry, setPipelineTelemetry] = useState<PipelineTelemetry | null>(null);
    const [telemetryHistory, setTelemetryHistory] = useState<PipelineTelemetry[]>([]);
    const [activeProviders, setActiveProviders] = useState<ActiveProviders | null>(null);
    const [discordBridge, setDiscordBridge] = useState<DiscordBridgeTelemetry | null>(null);
    const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
    const [pipelineStage, setPipelineStage] = useState<'idle' | 'stt' | 'queue' | 'llm' | 'tts' | 'speaking'>('idle');
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
    const [outputMode, setOutputMode] = useState<'discord' | 'local'>('discord');
    const [isSwitchingOutput, setIsSwitchingOutput] = useState(false);

    const [metrics, setMetrics] = useState({
        tps: 0,
        ttftMs: 0,
        e2eTtftMs: 0,
        sttLatencyMs: 0,
        ttsTtfaMs: 0,
        ttsStartMs: 0,
        e2eTtsStartMs: 0
    });

    const scrollRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const processedJobIds = useRef<Set<string>>(new Set());
    const persistTimerRef = useRef<number | null>(null);
    const pcmPlayer = useRef<PCMPlayer>(new PCMPlayer());

    // Обновляем мут-статус плеера при изменении isVoiceMuted
    useEffect(() => {
        pcmPlayer.current.setMuted(isVoiceMuted);
    }, [isVoiceMuted]);

    // Theme Switcher
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    useEffect(() => {
        wsClient.connect();
        const unsub = wsClient.subscribe((data) => {
            if (data.event === 'context_conversation_add_audio') {
                setPipelineStage('stt');
                return;
            }

            if (data.event === 'context_conversation_add_text') {
                const payload = data.payload;
                const result = payload?.result;

                if (result?.stt_latency_ms !== undefined) {
                    setMetrics(prev => ({ ...prev, sttLatencyMs: result.stt_latency_ms }));
                }
                setPipelineStage('queue');

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

            if (data.event === 'telemetry' || data.payload?.event === 'telemetry') {
                const tel: PipelineTelemetry = data.payload?.telemetry || data.payload;
                if (tel && tel.llm) {
                    setPipelineTelemetry(tel);
                    setTelemetryHistory(prev => [tel, ...prev.filter(x => x.job_id !== tel.job_id)].slice(0, 20));
                }
                return;
            }

            if (data.event === 'response') {
                if (data.payload?.start) {
                    setStatus('thinking');
                    setIsThinking(true);
                    setPipelineStage('llm');
                    if (data.payload?.stt_latency_ms) {
                        setMetrics(prev => ({ ...prev, sttLatencyMs: data.payload.stt_latency_ms }));
                    }
                    return;
                }

                const jobId = data.payload?.job_id;
                const result = data.payload?.result;

                // Глобальный/локальный stop сигнал (barge-in): просто выходим из thinking
                if (result?.event === 'stop_audio') {
                    pcmPlayer.current.stop(); // Обрываем текущий звук
                    setIsThinking(false);
                    setStatus('idle');
                    setPipelineStage('idle');
                    return;
                }

                // Штатная отмена job (не ошибка)
                if (result?.event === 'cancelled') {
                    setIsThinking(false);
                    setStatus('idle');
                    setPipelineStage('idle');

                    // Показываем системное сообщение только для ручной/внешней отмены
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
                    setPipelineStage('idle');

                    const tel: PipelineTelemetry = data.payload?.telemetry || data.payload?.result?.telemetry;
                    if (tel && tel.llm) {
                        setPipelineTelemetry(tel);
                        setTelemetryHistory(prev => [tel, ...prev.filter(x => x.job_id !== tel.job_id)].slice(0, 20));
                    }

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
                if (result.tps !== undefined) {
                    setMetrics(prev => ({
                        ...prev,
                        tps: result.tps ?? prev.tps,
                        ttftMs: result.llm_ttft_ms ?? result.ttft_ms ?? result.latency ?? prev.ttftMs,
                        e2eTtftMs: result.e2e_ttft_ms ?? prev.e2eTtftMs
                    }));
                }
                if (result.tts_start_ms !== undefined || result.e2e_tts_start_ms !== undefined || result.tts_ttfa_ms !== undefined) {
                    setMetrics(prev => ({
                        ...prev,
                        ttsTtfaMs: result.tts_ttfa_ms ?? prev.ttsTtfaMs,
                        ttsStartMs: result.tts_start_ms ?? prev.ttsStartMs,
                        e2eTtsStartMs: result.e2e_tts_start_ms ?? prev.e2eTtsStartMs
                    }));
                }

                if (content && !result.history) {
                    setMessages(prev => {
                        const existingIdx = prev.findIndex(m => m.id === jobId);
                        if (existingIdx !== -1) {
                            const newMessages = [...prev];
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
            
            if (data.event === 'audio_output_mode') {
                const mode = data.payload?.mode;
                if (mode === 'discord' || mode === 'local') {
                    setOutputMode(mode);
                }
            }

            // Воспроизведение аудио чанков от TTS
            if (data.event === 'audio_chunk') {
                setStatus('speaking');
                setPipelineStage('speaking');
                const payload = data.payload || data;
                if (payload.audio_bytes) {
                    // Локальное воспроизведение в браузере активно в режиме 'local' или если Discord не подключен к голосовому каналу
                    const shouldPlayLocal = outputMode === 'local' || !discordBridge?.connected_to_voice;
                    if (shouldPlayLocal && !isVoiceMuted) {
                        pcmPlayer.current.feedBase64(payload.audio_bytes, payload.sr || 44100);
                    }
                }
            }

        });

        loadHistory();

        // Поллинг телеметрии, провайдеров и статуса Discord моста
        const fetchPipeline = async () => {
            try {
                const res = await restClient.getPipeline();
                const data = res.response;
                if (data) {
                    if (data.active_providers) setActiveProviders(data.active_providers);
                    if (data.discord_bridge) setDiscordBridge(data.discord_bridge);
                    if (data.system) setSystemStats(data.system);
                    if (data.audio_output_mode && !isSwitchingOutput) {
                        setOutputMode(data.audio_output_mode);
                    }
                    if (data.telemetry) {
                        const newTel = data.telemetry;
                        setPipelineTelemetry(prev => prev?.job_id === newTel.job_id ? prev : newTel);
                    }
                    if (data.telemetry_history && data.telemetry_history.length > 0) {
                        setTelemetryHistory(data.telemetry_history);
                    }
                }
            } catch (e) { }
        };

        fetchPipeline();
        const pollInterval = window.setInterval(fetchPipeline, 2500);

        return () => {
            unsub();
            window.clearInterval(pollInterval);
        };
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
            await restClient.sendMessage(text, !isVoiceMuted);
        } catch (e) {
            setIsThinking(false);
            setStatus('idle');
            setMessages(prev => [...prev, { id: 'err-' + Date.now(), sender: 'Nira', text: 'LINK ERROR' }]);
        }
    };

    const handleClearTelemetry = async (e: React.MouseEvent) => {
        e.stopPropagation();
        setTelemetryHistory([]);
        setPipelineTelemetry(null);
        try {
            await restClient.clearTelemetry();
        } catch (err) {
            console.error('Failed to clear telemetry history:', err);
        }
    };

    const handleToggleOutputMode = async (targetMode?: 'discord' | 'local') => {
        const newMode = targetMode || (outputMode === 'discord' ? 'local' : 'discord');
        setOutputMode(newMode);
        setIsSwitchingOutput(true);
        try {
            await restClient.setOutputMode(newMode);
        } catch (err) {
            console.error('Failed to set output mode:', err);
        } finally {
            setTimeout(() => setIsSwitchingOutput(false), 600);
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

                    {/* Вывод звука: Discord vs Локально */}
                    <div className="output-mode-switch" title="Переключение вывода звука Ниры: Discord или Локально">
                        <button 
                            className={`mode-pill-btn ${outputMode === 'local' ? 'active' : ''}`}
                            onClick={() => handleToggleOutputMode('local')}
                            disabled={isSwitchingOutput}
                        >
                            <Headphones size={13} /> Локально
                        </button>
                        <button 
                            className={`mode-pill-btn discord ${outputMode === 'discord' ? 'active' : ''}`}
                            onClick={() => handleToggleOutputMode('discord')}
                            disabled={isSwitchingOutput}
                        >
                            <Radio size={13} /> Discord
                        </button>
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
                {/* LEFT: METRICS & TELEMETRY */}
                <aside className="pane">
                    <div className="header">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <BarChart3 size={18} color="var(--neon)" />
                            <span style={{ fontFamily: 'Orbitron', fontSize: '0.85rem', fontWeight: 700, letterSpacing: '1px' }}>
                                TELEMETRY
                            </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className="metric-badge accent" style={{ fontSize: '0.62rem' }}>
                                <span className={`indicator ${pipelineStage !== 'idle' ? 'on' : 'off'}`} style={{ width: '6px', height: '6px', marginRight: '4px' }} />
                                {pipelineStage.toUpperCase()}
                            </span>
                        </div>
                    </div>

                    <div className="pane-scrollable">
                        {/* 1. ACTIVE PROVIDERS STACK */}
                        <div>
                            <div className="section-title">
                                <span><Server size={12} style={{ display: 'inline', marginRight: '4px' }} /> Active Stack</span>
                                <span style={{ fontSize: '0.65rem', opacity: 0.6 }}>Local vs Cloud</span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', marginTop: '6px' }}>
                                {/* STT Provider Card */}
                                <div className="stat-card" style={{ padding: '8px 10px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                        <span className="stat-label" style={{ fontSize: '0.65rem', marginBottom: 0 }}>STT</span>
                                        <span className={`metric-badge ${activeProviders?.stt?.type === 'local' ? 'local' : 'cloud'}`}>
                                            {activeProviders?.stt?.type === 'local' ? 'Local' : 'Cloud'}
                                        </span>
                                    </div>
                                    <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={activeProviders?.stt?.id || 'stt'}>
                                        {activeProviders?.stt?.id || 'groq_stt'}
                                    </div>
                                    <div style={{ fontSize: '0.62rem', color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={activeProviders?.stt?.model || 'model'}>
                                        {activeProviders?.stt?.model || 'large-v3-turbo'}
                                    </div>
                                </div>

                                {/* LLM Provider Card */}
                                <div className="stat-card" style={{ padding: '8px 10px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                        <span className="stat-label" style={{ fontSize: '0.65rem', marginBottom: 0 }}>LLM</span>
                                        <span className={`metric-badge ${activeProviders?.t2t?.type === 'local' ? 'local' : 'cloud'}`}>
                                            {activeProviders?.t2t?.type === 'local' ? 'Local' : 'Cloud'}
                                        </span>
                                    </div>
                                    <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--neon)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={activeProviders?.t2t?.id || 'llm'}>
                                        {activeProviders?.t2t?.id || 'llamacpp'}
                                    </div>
                                    <div style={{ fontSize: '0.62rem', color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={activeProviders?.t2t?.model || 'model'}>
                                        {activeProviders?.t2t?.model || 'local'}
                                    </div>
                                </div>

                                {/* TTS Provider Card */}
                                <div className="stat-card" style={{ padding: '8px 10px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                        <span className="stat-label" style={{ fontSize: '0.65rem', marginBottom: 0 }}>TTS</span>
                                        <span className={`metric-badge ${activeProviders?.tts?.type === 'local' ? 'local' : 'cloud'}`}>
                                            {activeProviders?.tts?.type === 'local' ? 'Local' : 'Cloud'}
                                        </span>
                                    </div>
                                    <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--accent)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={activeProviders?.tts?.id || 'tts'}>
                                        {activeProviders?.tts?.id || 'fish_audio'}
                                    </div>
                                    <div style={{ fontSize: '0.62rem', color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={activeProviders?.tts?.model || 'model'}>
                                        {activeProviders?.tts?.model || 'fish-speech'}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* 2. LATENCY WATERFALL PIPELINE */}
                        <div>
                            <div className="section-title">
                                <span><Zap size={12} style={{ display: 'inline', marginRight: '4px' }} /> Latency Waterfall</span>
                                <span style={{ fontSize: '0.65rem', color: 'var(--neon)', fontFamily: 'Orbitron' }}>
                                    {(pipelineTelemetry?.total_latency_ms ?? pipelineTelemetry?.response_latency_ms)
                                        ? `${pipelineTelemetry?.total_latency_ms ?? pipelineTelemetry?.response_latency_ms}ms ${pipelineTelemetry?.input_mode === 'voice' ? 'VOICE E2E' : 'TOTAL E2E'}`
                                        : (pipelineTelemetry?.tts?.e2e_voice_start_ms || metrics.e2eTtsStartMs)
                                            ? `${pipelineTelemetry?.tts?.e2e_voice_start_ms || metrics.e2eTtsStartMs}ms VOICE E2E`
                                            : (metrics.ttftMs ? `${metrics.ttftMs}ms TTFT` : 'REALTIME')}
                                </span>
                            </div>

                            <div className="waterfall-track" style={{ marginTop: '6px' }}>
                                {/* Step 1: STT */}
                                <div className={`waterfall-step ${pipelineStage === 'stt' ? 'active' : (pipelineTelemetry?.stt?.latency_ms ? 'done' : '')}`}>
                                    <div className="waterfall-label">
                                        <Mic size={14} color={pipelineStage === 'stt' ? 'var(--neon)' : 'var(--text-dim)'} />
                                        <span>STT Voice In</span>
                                    </div>
                                    <div className="waterfall-time">
                                        {pipelineTelemetry?.stt?.latency_ms !== undefined && pipelineTelemetry?.stt?.latency_ms !== null
                                            ? `${pipelineTelemetry.stt.latency_ms}ms`
                                            : (metrics.sttLatencyMs ? `${metrics.sttLatencyMs}ms` : '—')}
                                    </div>
                                </div>

                                {/* Step 2: Queue Overhead */}
                                <div className={`waterfall-step ${pipelineStage === 'queue' ? 'active' : (pipelineTelemetry?.queue_overhead_ms !== undefined ? 'done' : '')}`}>
                                    <div className="waterfall-label">
                                        <Clock size={14} color={pipelineStage === 'queue' ? 'var(--neon)' : 'var(--text-dim)'} />
                                        <span>Queue / Pre-LLM</span>
                                    </div>
                                    <div className="waterfall-time" style={{ color: 'var(--text-dim)' }}>
                                        {pipelineTelemetry?.queue_overhead_ms !== undefined ? `${pipelineTelemetry.queue_overhead_ms}ms` : '—'}
                                    </div>
                                </div>

                                {/* Step 3: LLM TTFT */}
                                <div className={`waterfall-step ${pipelineStage === 'llm' ? 'active' : (pipelineTelemetry?.llm?.ttft_ms ? 'done' : '')}`}>
                                    <div className="waterfall-label">
                                        <Brain size={14} color={pipelineStage === 'llm' ? 'var(--neon)' : 'var(--text-dim)'} />
                                        <span>LLM TTFT</span>
                                    </div>
                                    <div className="waterfall-time" style={{ color: 'var(--neon)' }}>
                                        {(pipelineTelemetry?.llm?.ttft_ms ?? metrics.ttftMs ?? 0)}ms
                                        {(pipelineTelemetry?.llm?.e2e_ttft_ms || metrics.e2eTtftMs) ? (
                                            <span style={{ fontSize: '0.62rem', color: 'var(--text-dim)', marginLeft: '4px' }}>
                                                ({pipelineTelemetry?.llm?.e2e_ttft_ms || metrics.e2eTtftMs}ms e2e)
                                            </span>
                                        ) : null}
                                    </div>
                                </div>

                                {/* Step 4: LLM TPS & Stream */}
                                <div className={`waterfall-step ${pipelineStage === 'llm' ? 'active' : ''}`}>
                                    <div className="waterfall-label">
                                        <Gauge size={14} color="var(--text-dim)" />
                                        <span>Tokens / Sec</span>
                                    </div>
                                    <div className="waterfall-time" style={{ color: '#10b981' }}>
                                        {(metrics.tps || pipelineTelemetry?.llm?.tps || 0)} <span style={{ fontSize: '0.62rem', color: 'var(--text-dim)' }}>tps</span>
                                        {pipelineTelemetry?.llm?.token_count ? (
                                            <span style={{ fontSize: '0.62rem', color: 'var(--text-dim)', marginLeft: '4px' }}>
                                                ({pipelineTelemetry.llm.token_count} tok)
                                            </span>
                                        ) : null}
                                    </div>
                                </div>

                                {/* Step 5: TTS Dispatch */}
                                <div className={`waterfall-step ${pipelineTelemetry?.llm?.first_sentence_ms ? 'done' : ''}`}>
                                    <div className="waterfall-label">
                                        <Layers size={14} color="var(--text-dim)" />
                                        <span>TTS Phrase Ready</span>
                                    </div>
                                    <div className="waterfall-time" style={{ color: 'var(--text-dim)' }}>
                                        {pipelineTelemetry?.llm?.first_sentence_ms !== undefined && pipelineTelemetry?.llm?.first_sentence_ms !== null
                                            ? `${pipelineTelemetry.llm.first_sentence_ms}ms`
                                            : '—'}
                                    </div>
                                </div>

                                {/* Step 6: TTS TTFA */}
                                <div className={`waterfall-step ${pipelineStage === 'tts' || pipelineStage === 'speaking' ? 'active' : (pipelineTelemetry?.tts?.ttfa_ms ? 'done' : '')}`}>
                                    <div className="waterfall-label">
                                        <Volume2 size={14} color={pipelineStage === 'speaking' ? 'var(--accent)' : 'var(--text-dim)'} />
                                        <span>TTS Audio Start (TTFA)</span>
                                    </div>
                                    <div className="waterfall-time" style={{ color: 'var(--accent)' }}>
                                        {(pipelineTelemetry?.tts?.ttfa_ms ?? metrics.ttsTtfaMs ?? metrics.ttsStartMs)
                                            ? `${pipelineTelemetry?.tts?.ttfa_ms ?? metrics.ttsTtfaMs ?? metrics.ttsStartMs}ms`
                                            : '—'}
                                    </div>
                                </div>

                                {/* Step 7: First Audio Out / E2E */}
                                <div className="waterfall-step" style={{ background: 'rgba(56, 189, 248, 0.05)', borderColor: 'var(--border-nira)' }}>
                                    <div className="waterfall-label">
                                        <CheckCircle2 size={14} color="var(--accent)" />
                                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                                            <span style={{ fontWeight: 700 }}>
                                                {pipelineTelemetry?.input_mode === 'voice' ? 'Общая задержка (конец речи → звук)' : 'Общая задержка (отправка → звук)'}
                                            </span>
                                            {pipelineTelemetry?.input_mode === 'voice' && pipelineTelemetry.user_speech_duration_ms !== undefined && (
                                                <span style={{ fontSize: '0.60rem', color: 'var(--text-dim)' }}>
                                                    Длительность речи: {pipelineTelemetry.user_speech_duration_ms}ms · До первого звука: {pipelineTelemetry.turn_taking_latency_ms || pipelineTelemetry.total_latency_ms}ms
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    <div className="waterfall-time" style={{ color: 'var(--accent)', fontSize: '0.88rem' }}>
                                        {(pipelineTelemetry?.total_latency_ms ?? pipelineTelemetry?.response_latency_ms ?? pipelineTelemetry?.tts?.e2e_voice_start_ms ?? pipelineTelemetry?.tts?.first_audio_ms ?? metrics.e2eTtsStartMs)
                                            ? `${pipelineTelemetry?.total_latency_ms ?? pipelineTelemetry?.response_latency_ms ?? pipelineTelemetry?.tts?.e2e_voice_start_ms ?? pipelineTelemetry?.tts?.first_audio_ms ?? metrics.e2eTtsStartMs}ms`
                                            : '—'}
                                        {pipelineTelemetry?.total_pipeline_ms ? (
                                            <span style={{ fontSize: '0.60rem', color: 'var(--text-dim)', marginLeft: '6px', fontWeight: 'normal' }}>
                                                (все фразы: {(pipelineTelemetry.total_pipeline_ms / 1000).toFixed(1)}s)
                                            </span>
                                        ) : null}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* 3. DISCORD BRIDGE TELEMETRY */}
                        <div className="discord-status-card">
                            <div className="discord-header">
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <Radio size={15} color="#5865f2" />
                                    <span>DISCORD BRIDGE</span>
                                </div>
                                <span className={`metric-badge ${discordBridge?.online ? (discordBridge.connected_to_voice ? 'local' : 'accent') : 'warning'}`}>
                                    {discordBridge?.online ? (discordBridge.connected_to_voice ? 'IN VOICE' : 'ONLINE') : 'OFFLINE'}
                                </span>
                            </div>

                            {discordBridge?.connected_to_voice && discordBridge.channel_name && (
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    <span>🔊 Канал:</span>
                                    <span style={{ color: 'var(--text)', fontWeight: 600 }}>{discordBridge.channel_name}</span>
                                </div>
                            )}

                            <div className="discord-pings">
                                <div className="ping-pill">
                                    <span className="ping-pill-title">Gateway WS</span>
                                    <span className="ping-pill-val" style={{ color: discordBridge?.gateway_ping_ms && discordBridge.gateway_ping_ms < 100 ? '#10b981' : '#f59e0b' }}>
                                        {discordBridge?.gateway_ping_ms !== undefined && discordBridge.gateway_ping_ms !== null ? `${discordBridge.gateway_ping_ms} ms` : '—'}
                                    </span>
                                </div>
                                <div className="ping-pill">
                                    <span className="ping-pill-title">Voice UDP</span>
                                    <span className="ping-pill-val" style={{ color: discordBridge?.voice_ping_ms && discordBridge.voice_ping_ms < 80 ? '#10b981' : '#f59e0b' }}>
                                        {discordBridge?.voice_ping_ms !== undefined && discordBridge.voice_ping_ms !== null ? `${discordBridge.voice_ping_ms} ms` : '—'}
                                    </span>
                                </div>
                            </div>

                            {/* Кнопка быстрого переключения Discord канала на лету */}
                            <button 
                                className={`discord-quick-toggle ${discordBridge?.connected_to_voice ? 'leave' : 'join'}`}
                                onClick={() => handleToggleOutputMode(discordBridge?.connected_to_voice ? 'local' : 'discord')}
                                disabled={isSwitchingOutput || !discordBridge?.online}
                                title={discordBridge?.connected_to_voice 
                                    ? "Отключить Ниру от голосового канала (звук пойдет локально в наушники)" 
                                    : "Подключить Ниру к голосовому каналу Discord"}
                            >
                                {discordBridge?.connected_to_voice ? (
                                    <>
                                        <Headphones size={13} /> Отключить из канала → Наушники
                                    </>
                                ) : (
                                    <>
                                        <Radio size={13} /> Подключить к каналу Discord
                                    </>
                                )}
                            </button>
                        </div>

                        {/* 4. RUNS COMPARISON LOG */}
                        <div>
                            <div className="section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span><Activity size={12} style={{ display: 'inline', marginRight: '4px' }} /> Сравнение провайдеров</span>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <span style={{ fontSize: '0.65rem', opacity: 0.6 }}>{telemetryHistory.length} прогонов</span>
                                    {telemetryHistory.length > 0 && (
                                        <button 
                                            onClick={handleClearTelemetry}
                                            title="Очистить историю сравнения"
                                            style={{ 
                                                background: 'transparent', 
                                                border: 'none', 
                                                cursor: 'pointer', 
                                                color: 'var(--text-dim)', 
                                                fontSize: '0.65rem',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '3px',
                                                padding: '2px 6px',
                                                borderRadius: '4px',
                                            }}
                                            onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
                                            onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-dim)')}
                                        >
                                            <Trash2 size={11} /> Очистить
                                        </button>
                                    )}
                                </div>
                            </div>

                            <div className="runs-list" style={{ marginTop: '6px' }}>
                                {telemetryHistory.length === 0 ? (
                                    <div style={{ fontSize: '0.73rem', color: 'var(--text-dim)', padding: '10px', textAlign: 'center', fontStyle: 'italic' }}>
                                        Лог прогонов пуст. Отправьте сообщение или скажите фразу в микрофон / Discord.
                                    </div>
                                ) : (
                                    telemetryHistory.map((run, idx) => {
                                        const runLatency = run.total_latency_ms
                                            ?? run.response_latency_ms 
                                            ?? run.tts?.e2e_voice_start_ms 
                                            ?? run.tts?.first_audio_ms 
                                            ?? run.llm?.ttft_ms;
                                        return (
                                            <div
                                                key={run.job_id || idx}
                                                className="run-row"
                                                onClick={() => setSelectedRunId(selectedRunId === run.job_id ? null : (run.job_id || null))}
                                                style={{ cursor: 'pointer', borderColor: selectedRunId === run.job_id ? 'var(--neon)' : undefined }}
                                            >
                                                <div className="run-row-header">
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px', flexWrap: 'wrap' }}>
                                                        <span className={`metric-badge ${run.llm?.type === 'local' ? 'local' : 'cloud'}`} style={{ fontSize: '0.58rem', padding: '1px 5px' }}>
                                                            LLM: {run.llm?.provider || 'llm'}
                                                        </span>
                                                        <span className={`metric-badge ${run.tts?.type === 'local' ? 'local' : 'cloud'}`} style={{ fontSize: '0.58rem', padding: '1px 5px' }}>
                                                            TTS: {run.tts?.provider || 'tts'}
                                                        </span>
                                                        <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)', fontWeight: 600 }}>
                                                            {run.input_mode === 'voice' ? '🎤 Голос' : '💬 Текст'}
                                                        </span>
                                                    </div>
                                                    <div style={{ textAlign: 'right' }}>
                                                        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'flex-end', gap: '4px' }}>
                                                            <span style={{ fontFamily: 'Orbitron', fontSize: '0.74rem', color: 'var(--neon)', fontWeight: 700 }}>
                                                                {runLatency ? `${runLatency}ms` : (run.total_pipeline_ms ? `${run.total_pipeline_ms}ms` : '—')}
                                                            </span>
                                                        </div>
                                                        <div style={{ fontSize: '0.56rem', color: 'var(--text-dim)' }}>
                                                            {run.input_mode === 'voice' ? 'конец речи → звук' : 'отправка → звук'}
                                                        </div>
                                                        {run.total_pipeline_ms && runLatency && run.total_pipeline_ms > runLatency * 1.5 ? (
                                                            <div style={{ fontSize: '0.54rem', color: 'var(--text-dim)' }}>
                                                                все фразы: {(run.total_pipeline_ms / 1000).toFixed(1)}s
                                                            </div>
                                                        ) : null}
                                                    </div>
                                                </div>

                                                <div className="run-row-stats">
                                                    {run.stt?.latency_ms ? <span>STT: <b style={{ color: 'var(--text)' }}>{run.stt.latency_ms}ms</b></span> : null}
                                                    <span>TTFT: <b style={{ color: 'var(--text)' }}>{run.llm?.ttft_ms}ms</b></span>
                                                    <span>TPS: <b style={{ color: '#10b981' }}>{run.llm?.tps || '—'}</b></span>
                                                    <span>TTS: <b style={{ color: 'var(--accent)' }}>{run.tts?.ttfa_ms ? `${run.tts.ttfa_ms}ms` : '—'}</b></span>
                                                </div>

                                                {selectedRunId === run.job_id && (
                                                    <div style={{ marginTop: '5px', paddingTop: '5px', borderTop: '1px dashed var(--border)', fontSize: '0.68rem', color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                                                        <div style={{ color: 'var(--neon)', fontWeight: 600 }}>
                                                            ⏱ Точки замера: {run.input_mode === 'voice' ? 'Последний пакет голоса (конец речи)' : 'Отправка сообщения в чат'} → Начало воспроизведения звука ({runLatency}ms)
                                                        </div>
                                                        {run.input_mode === 'voice' && run.user_speech_duration_ms !== undefined && (
                                                            <div>Длительность речи: <b>{run.user_speech_duration_ms}ms</b> · До первого звука: <b>{run.turn_taking_latency_ms || runLatency}ms</b></div>
                                                        )}
                                                        <div>STT: <b>{run.stt?.provider || 'none'}</b> ({run.stt?.model || '—'}) {run.stt?.latency_ms ? `· ${run.stt.latency_ms}ms` : ''}</div>
                                                        <div>LLM: <b>{run.llm?.provider}</b> ({run.llm?.model || '—'}) · {run.llm?.token_count || 0} токенов</div>
                                                        <div>TTS: <b>{run.tts?.provider}</b> ({run.tts?.model || '—'}) · TTFA {run.tts?.ttfa_ms}ms</div>
                                                        {run.total_pipeline_ms && <div>Полный цикл синтеза всех фраз: {(run.total_pipeline_ms / 1000).toFixed(2)}s</div>}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })
                                )}
                            </div>
                        </div>

                        {/* 5. HARDWARE DUAL GPU MONITOR */}
                        {systemStats && (
                            <div className="hw-card">
                                <div className="section-title" style={{ marginBottom: 0 }}>
                                    <span><Cpu size={12} style={{ display: 'inline', marginRight: '4px' }} /> Hardware Load</span>
                                    <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>RTX 5000 Series</span>
                                </div>

                                {systemStats.gpus && systemStats.gpus.map((gpu) => {
                                    const vramPct = Math.round((gpu.mem_used / Math.max(1, gpu.mem_total)) * 100);
                                    return (
                                        <div key={gpu.id} className="hw-item">
                                            <div className="hw-header">
                                                <span style={{ fontWeight: 600, color: 'var(--text)' }}>
                                                    GPU {gpu.id}: {gpu.name.replace('NVIDIA GeForce ', '')}
                                                </span>
                                                <span style={{ fontFamily: 'Orbitron', color: gpu.load > 75 ? '#f59e0b' : 'var(--neon)' }}>
                                                    {gpu.load}% · {gpu.temp}°C
                                                </span>
                                            </div>
                                            <div className="hw-bar-wrapper">
                                                <div
                                                    className="hw-bar-fill"
                                                    style={{
                                                        width: `${gpu.load}%`,
                                                        background: gpu.load > 85 ? '#ef4444' : (gpu.load > 50 ? '#f59e0b' : 'var(--neon)')
                                                    }}
                                                />
                                            </div>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--text-dim)' }}>
                                                <span>VRAM: {gpu.mem_used} / {gpu.mem_total} MB</span>
                                                <span>{vramPct}%</span>
                                            </div>
                                        </div>
                                    );
                                })}

                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '2px', paddingTop: '4px', borderTop: '1px solid var(--border)' }}>
                                    <div style={{ fontSize: '0.7rem' }}>
                                        <span style={{ color: 'var(--text-dim)' }}>CPU: </span>
                                        <b style={{ color: 'var(--text)' }}>{systemStats.cpu}%</b>
                                    </div>
                                    <div style={{ fontSize: '0.7rem' }}>
                                        <span style={{ color: 'var(--text-dim)' }}>RAM: </span>
                                        <b style={{ color: 'var(--text)' }}>{systemStats.ram}%</b>
                                    </div>
                                </div>
                            </div>
                        )}
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
                                        <div className="msg-info" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                            <span>{m.sender === 'Creator' ? 'CREATOR' : 'NIRA_AI'}</span>
                                            {m.sourceId && (
                                                <span style={{ opacity: 0.6, fontSize: '0.62rem', fontWeight: 'normal' }}>
                                                    {m.sourceId === 'discord' ? '🎧 Discord Voice' : m.sourceId === 'mic' ? '🎤 Mic' : '💬 Web'}
                                                </span>
                                            )}
                                        </div>
                                        <div>{m.text}</div>
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
