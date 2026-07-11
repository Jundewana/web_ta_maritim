import { useState, useRef, useEffect, useCallback } from "react";
import "./index.css";
import {
  UploadCloud, FileText, X, Send, Trash2, Clock,
  ChevronDown, ChevronRight, FolderOpen, Plus, MessageSquare,
  Zap, Database, CheckCircle2, Loader2, BarChart2, AlertCircle,
  Sun, Moon,
} from "lucide-react";

/* ─── Constants ───────────────────────────────────────────────────── */
const MODES = {
  naive: {
    key: "naive", label: "NaiveRAG", tagline: "Vector-based retrieval",
    avatar: "N",
    icon: (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>),
  },
  graph: {
    key: "graph", label: "GraphRAG", tagline: "Knowledge graph retrieval",
    avatar: "G",
    icon: (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><line x1="12" y1="7" x2="5" y2="17"/><line x1="12" y1="7" x2="19" y2="17"/><line x1="7" y1="19" x2="17" y2="19"/></svg>),
  },
};

/* Dark-mode's vivid teal/purple (#00C9A7 / #A78BFA) pop against the
   near-black canvas but drop to ~2:1 contrast on the light theme's cream/
   white surfaces -- barely legible as text. Light mode gets deeper steps of
   the same hue instead of the same literal hex, so identity is preserved
   but text/icon contrast clears WCAG AA (~4.5:1) on both grounds. Colors
   are applied as inline-style hex strings all over this file (many with a
   "+alpha suffix" trick, e.g. accent+"66"), so the fix has to happen at
   this resolution layer rather than via CSS custom properties alone. */
const ACCENTS = {
  naive: { dark: "#00C9A7", light: "#00806B" },
  graph: { dark: "#A78BFA", light: "#7C3AED" },
};
function accentFor(key, theme) {
  return ACCENTS[key]?.[theme] ?? ACCENTS[key]?.dark;
}
function modeFor(key, theme) {
  return { ...MODES[key], color: accentFor(key, theme) };
}

const STATUS = {
  good: { dark: "#00C9A7", light: "#00806B" },
  warn: { dark: "#F59E0B", light: "#B45309" },
  bad:  { dark: "#F87171", light: "#DC2626" },
};
function statusColor(level, theme) {
  return STATUS[level][theme] ?? STATUS[level].dark;
}
// Dark mode's accents are bright (near-black ink reads well on them); light
// mode's are deepened for text contrast (see ACCENTS above), which flips the
// legible ink to white -- used for avatar letters / send-button icon, which
// sit ON a solid accent-colored fill rather than the page surface.
function inkOnAccent(theme) {
  return theme === "light" ? "#FFFFFF" : "#0D0F14";
}

const FOLDERS = ["Pelayaran", "Perikanan"];

/* ─── Helpers ─────────────────────────────────────────────────────── */
function genId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }
function save(s) { try { localStorage.setItem("mc_sessions_v3", JSON.stringify(s)); } catch {} }
function load()  { try { return JSON.parse(localStorage.getItem("mc_sessions_v3") || "[]"); } catch { return []; } }

function loadTheme() {
  try {
    const stored = localStorage.getItem("mc_theme");
    if (stored === "light" || stored === "dark") return stored;
  } catch {}
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function sessionTitle(msgs) {
  const f = msgs.find(m => m.role === "user");
  if (!f) return "Percakapan baru";
  return f.content.slice(0, 46) + (f.content.length > 46 ? "…" : "");
}

/**
 * Mode dominan sesi — untuk sesi KOSONG pakai createdMode.
 * Ini fix bug "Chat Baru selalu masuk GraphRAG".
 */
function sessionDominantMode(s) {
  const msgs = s.messages || [];
  const userMsgs = msgs.filter(m => m.role === "user" && m.msgMode);
  if (userMsgs.length === 0) return s.createdMode || "naive"; // ← fix
  const counts = { naive: 0, graph: 0 };
  userMsgs.forEach(m => { if (counts[m.msgMode] !== undefined) counts[m.msgMode]++; });
  return counts.naive >= counts.graph ? "naive" : "graph";
}

function lastMsgMode(msgs) {
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === "user" && msgs[i].msgMode) return msgs[i].msgMode;
  }
  return null;
}

/* ─── Score bar component ─────────────────────────────────────────── */
function ScoreBar({ label, score, theme }) {
  const pct = Math.round((score ?? 0) * 100);
  const barColor = pct >= 70 ? statusColor("good", theme) : pct >= 40 ? statusColor("warn", theme) : statusColor("bad", theme);
  return (
    <div className="score-row">
      <div className="score-label">{label}</div>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${pct}%`, background: barColor }}/>
      </div>
      <div className="score-value" style={{ color: barColor }}>{pct}%</div>
    </div>
  );
}

/* ─── RAGAS Panel ─────────────────────────────────────────────────── */
/* Presentational only -- state lives in App so it survives the panel
   being hidden/shown (previously: closing the panel unmounted this
   component and threw away in-flight "checking"/"running" state, so
   reopening it looked like nothing had happened). */
function RagasPanel({ lastUserQuery, status, matchInfo, scores, errMsg, onRunEval, mode, theme }) {
  const m = modeFor(mode, theme);

  return (
    <div className="ragas-panel">
      <div className="ragas-header">
        <BarChart2 size={13} style={{ color: m.color }}/>
        <span className="ragas-title">RAGAS Evaluasi</span>
      </div>

      {!lastUserQuery && (
        <p className="ragas-hint">Kirim pesan dulu untuk mengevaluasi jawaban.</p>
      )}

      {lastUserQuery && status === "idle" && (
        <>
          <p className="ragas-query-preview">"{lastUserQuery.slice(0, 60)}{lastUserQuery.length > 60 ? "…" : ""}"</p>
          <button className="ragas-btn" style={{ borderColor: m.color+"66", color: m.color }} onClick={onRunEval}>
            <BarChart2 size={12}/> Cek Ground Truth & Evaluasi
          </button>
        </>
      )}

      {status === "checking" && (
        <div className="ragas-loading"><Loader2 size={13} className="spin"/> Mencari ground truth di dataset…</div>
      )}

      {status === "no_match" && (
        <div className="ragas-nomatch">
          <AlertCircle size={13}/>
          <span>Pertanyaan tidak ditemukan di dataset QA. Coba pertanyaan lain dari CSV.</span>
        </div>
      )}

      {status === "running" && (
        <div className="ragas-loading">
          <Loader2 size={13} className="spin"/>
          <div>
            <div>Menjalankan evaluasi RAGAS…</div>
            {matchInfo && <div className="ragas-match-info">GT dari: {matchInfo.dataset}</div>}
          </div>
        </div>
      )}

      {status === "error" && (
        <div className="ragas-error"><AlertCircle size={13}/> {errMsg}</div>
      )}

      {status === "done" && scores && (
        <div className="ragas-results">
          {matchInfo && (
            <div className="ragas-gt-box">
              <div className="ragas-gt-label">Ground Truth ({matchInfo.dataset})</div>
              <div className="ragas-gt-text">{matchInfo.ground_truth.slice(0, 150)}{matchInfo.ground_truth.length > 150 ? "…" : ""}</div>
            </div>
          )}
          {/* Info context yang dipakai */}
          <div className={`ragas-ctx-info ${scores.context_available ? "ctx-ok" : "ctx-warn"}`}>
            {scores.context_available
              ? `✓ Context tersedia (${scores.context_chars} karakter)`
              : "⚠ Context tidak tersedia — Faithfulness & Context Recall = 0"}
          </div>
          <div className="ragas-scores">
            <ScoreBar label="Ans Correctness" score={scores.answer_correctness} theme={theme}/>
            <ScoreBar label="Faithfulness"    score={scores.faithfulness} theme={theme}/>
            <ScoreBar label="Context Recall"  score={scores.context_recall} theme={theme}/>
          </div>
          <button className="ragas-btn ragas-btn-rerun" onClick={onRunEval}>
            ↺ Ulangi
          </button>
        </div>
      )}
    </div>
  );
}

/* ─── TypingDots ──────────────────────────────────────────────────── */
function TypingDots({ color }) {
  return (
    <div className="typing-dots">
      {[0,1,2].map(i => <span key={i} style={{ background: color, animationDelay:`${i*0.18}s` }}/>)}
    </div>
  );
}

/* ─── Message ─────────────────────────────────────────────────────── */
function Message({ msg, theme }) {
  const modeKey = msg.msgMode || "naive";
  const m       = modeFor(modeKey, theme);
  const isUser  = msg.role === "user";
  return (
    <div className={`msg-row ${isUser ? "msg-user" : "msg-bot"}`}>
      {!isUser && (
        <div className="msg-avatar" style={{ background: m.color, color: inkOnAccent(theme) }}>{m.avatar}</div>
      )}
      <div
        className={`msg-bubble ${isUser ? "bubble-user" : "bubble-bot"}`}
        style={!isUser ? { borderColor: m.color + "40" } : {}}
      >
        {msg.content}
        {msg.role === "user" && (
          <div className="msg-mode-badge" style={{ color: m.color }}>{m.label}</div>
        )}
        {msg.kb && (
          <div className="msg-kb-tag"><Database size={10}/> {msg.kb}</div>
        )}
      </div>
    </div>
  );
}

/* ─── PickerModal ─────────────────────────────────────────────────── */
function PickerModal({ mode, theme, onConfirm, onClose }) {
  const [folder, setFolder]     = useState(null);
  const [items, setItems]       = useState([]);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const isNaive     = mode === "naive";
  const accentColor = accentFor(mode, theme);

  const openFolder = async (name) => {
    setFolder(name); setItems([]); setSelected([]); setError(null); setLoading(true);
    try {
      const res  = await fetch(`/api/dataset/list?folder=${encodeURIComponent(name)}&mode=${mode}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Gagal memuat daftar");
      setItems(data.items || []);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const toggle = (item) =>
    setSelected(p => p.includes(item) ? p.filter(x => x !== item) : [...p, item]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-row">
            {isNaive ? <><FileText size={15}/> Pilih PDF — NaiveRAG</>
                     : <><Database size={15}/> Pilih Knowledge Base — GraphRAG</>}
          </div>
          <button className="modal-close" onClick={onClose}><X size={15}/></button>
        </div>
        <div className="modal-body">
          <div className="folder-tabs">
            {FOLDERS.map(name => (
              <button key={name}
                className={`folder-tab ${folder === name ? "folder-tab-active" : ""}`}
                style={folder === name ? { background: accentColor+"1A", borderColor: accentColor+"66", color: accentColor } : {}}
                onClick={() => openFolder(name)}>
                <FolderOpen size={13}/> {name}
              </button>
            ))}
          </div>
          {!loading && !error && items.length > 0 && (
            <div className="picker-toolbar">
              <span className="picker-multisel-hint">
                {selected.length > 0 ? `${selected.length} / ${items.length} dipilih` : "Pilih satu atau lebih"}
              </span>
              <div style={{ display:"flex", gap:6 }}>
                <button className="picker-action-btn" onClick={() => setSelected([...items])}>Semua</button>
                <button className="picker-action-btn" onClick={() => setSelected([])}>Reset</button>
              </div>
            </div>
          )}
          <div className="file-list">
            {!folder  && <p className="picker-hint">Pilih folder di atas</p>}
            {loading  && <p className="picker-hint">Memuat…</p>}
            {error    && <p className="picker-error">{error}</p>}
            {!loading && !error && items.map(item => {
              const checked = selected.includes(item);
              return (
                <button key={item}
                  className={`file-item ${checked ? "file-item-selected" : ""}`}
                  style={checked ? { borderColor: accentColor+"44", background: accentColor+"0D" } : {}}
                  onClick={() => toggle(item)}>
                  {isNaive
                    ? <FileText size={13} style={{ flexShrink:0, color: checked ? accentColor : "rgba(var(--ink-rgb),0.3)" }}/>
                    : <Database size={13} style={{ flexShrink:0, color: checked ? accentColor : "rgba(var(--ink-rgb),0.3)" }}/>}
                  <span className="file-name">{item}</span>
                  {checked && <CheckCircle2 size={13} style={{ flexShrink:0, color: accentColor }}/>}
                </button>
              );
            })}
            {!loading && !error && folder && items.length === 0 && (
              <p className="picker-hint">Tidak ada item ditemukan</p>
            )}
          </div>
        </div>
        <div className="modal-footer">
          <span className="modal-sel-count">
            {selected.length > 0 ? `${selected.length} item dipilih` : "Belum ada yang dipilih"}
          </span>
          <div style={{ display:"flex", gap:8 }}>
            <button className="modal-btn modal-btn-cancel" onClick={onClose}>Batal</button>
            <button className="modal-btn modal-btn-confirm"
              disabled={selected.length === 0 || !folder}
              style={{ background: selected.length > 0 && folder ? accentColor : undefined }}
              onClick={() => onConfirm(folder, selected)}>
              {isNaive ? `Indeks ${selected.length || ""} PDF` : `Gunakan ${selected.length || ""} KB`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── KBPanel ─────────────────────────────────────────────────────── */
function KBPanel({ mode, theme, kbState, onUpload, onOpenPicker, onClear, fileInputRef }) {
  const isNaive = mode === "naive";
  const color   = accentFor(mode, theme);
  const { label, status, busy } = kbState;

  if (busy) return (
    <div className="sidebar-section">
      <p className="nav-label">Knowledge Base</p>
      <div className="kb-busy"><Loader2 size={14} className="spin" style={{ color }}/><span>Mengindeks…</span></div>
    </div>
  );

  return (
    <div className="sidebar-section">
      <p className="nav-label">Knowledge Base</p>
      {label ? (
        <div className="kb-active" style={{ borderColor: color+"44", background: color+"0D" }}>
          {isNaive ? <FileText size={14} color={color}/> : <Database size={14} color={color}/>}
          <div className="upload-info">
            <span className="upload-name" title={label}>{label}</span>
            <span className="upload-status" style={{ color }}>{status}</span>
          </div>
          <button className="upload-remove" onClick={onClear}><X size={12}/></button>
        </div>
      ) : isNaive ? (
        <div className="kb-naive-actions">
          <label className="upload-area">
            <input type="file" accept=".pdf" onChange={onUpload} ref={fileInputRef} style={{ display:"none" }}/>
            <UploadCloud size={18} color={color}/>
            <span className="upload-text">Upload PDF</span>
            <span className="upload-sub">Klik untuk memilih file</span>
          </label>
          <button className="dataset-pick-btn" onClick={onOpenPicker} style={{ borderColor: color+"44" }}>
            <FolderOpen size={13} color={color}/><span>Pilih dari Dataset</span>
          </button>
        </div>
      ) : (
        <div className="kb-naive-actions">
          <button className="dataset-pick-btn" onClick={onOpenPicker} style={{ borderColor: color+"44" }}>
            <Database size={13} color={color}/><span>Pilih Knowledge Base</span>
          </button>
          <p className="kb-auto-hint"><Zap size={10}/> Auto-detect jika tidak dipilih</p>
        </div>
      )}
    </div>
  );
}

/* ─── HistorySection ─────────────────────────────────────────────── */
function HistorySection({ modeKey, sessions, activeId, onLoad, onDelete, theme }) {
  const [open, setOpen] = useState(true);
  const mod = modeFor(modeKey, theme);
  const filtered = sessions.filter(s => sessionDominantMode(s) === modeKey);
  if (filtered.length === 0) return null;
  return (
    <div className="history-group">
      <button className="history-toggle" onClick={() => setOpen(o => !o)}>
        <span style={{ color: mod.color, display:"flex", alignItems:"center" }}>{mod.icon}</span>
        <span className="history-group-label" style={{ color: mod.color }}>{mod.label}</span>
        <span className="history-count">{filtered.length}</span>
        {open ? <ChevronDown size={10}/> : <ChevronRight size={10}/>}
      </button>
      {open && (
        <div className="history-list">
          {filtered.map(s => (
            <button key={s.id}
              className={`history-item ${s.id === activeId ? "history-active" : ""}`}
              style={s.id === activeId ? { borderColor: mod.color+"44", color: mod.color } : {}}
              onClick={() => onLoad(s)}
            >
              <MessageSquare size={10} style={{ flexShrink:0, opacity:0.5 }}/>
              <div className="history-info">
                <span className="history-title">{sessionTitle(s.messages)}</span>
                <span className="history-meta">
                  {new Date(s.ts).toLocaleDateString("id-ID",{day:"2-digit",month:"short"})}
                </span>
              </div>
              <button className="history-del" onClick={e => onDelete(s.id, e)}><X size={10}/></button>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── App ─────────────────────────────────────────────────────────── */
export default function App() {
  const [mode, setMode]               = useState("naive");
  const [theme, setTheme]             = useState(() => loadTheme());
  const [sessions, setSessions]       = useState(() => load());
  const [activeId, setActiveId]       = useState(null);
  const [modeActiveId, setModeActiveId] = useState({ naive:null, graph:null });
  const [input, setInput]             = useState("");
  const [typingSessions, setTypingSessions] = useState(() => new Set());
  const [historyOpen, setHistoryOpen] = useState(true);
  const [showPicker, setShowPicker]   = useState(false);
  const [showRagas, setShowRagas]     = useState(false);

  const [naiveKb, setNaiveKb] = useState({ label:"", status:"", busy:false, folder:null, filenames:[] });
  const [graphKb, setGraphKb] = useState({ label:"", status:"", busy:false, folder:null, kbNames:[], kbPaths:[] });

  // Track last exchange for RAGAS
  const [lastUserQuery, setLastUserQuery]   = useState("");
  const [lastBotResponse, setLastBotResponse] = useState("");
  const [lastContext, setLastContext]         = useState("");
  // Unique id of the bot message the above three describe -- used (instead
  // of lastUserQuery's *text*) to key the RAGAS reset below, since two
  // different sessions can easily share the exact same question text; text
  // alone wouldn't change and the reset effect would never fire, leaving
  // session A's stale RAGAS result showing under session B's question.
  const [lastBotMsgId, setLastBotMsgId]       = useState(null);

  // RAGAS eval state -- lives here (not inside RagasPanel) so an in-flight
  // "checking"/"running" evaluation survives the panel being collapsed and
  // reopened, instead of being thrown away when RagasPanel unmounts.
  const [ragasStatus, setRagasStatus]     = useState("idle"); // idle | checking | running | done | error | no_match
  const [ragasMatchInfo, setRagasMatchInfo] = useState(null);
  const [ragasScores, setRagasScores]     = useState(null);
  const [ragasErrMsg, setRagasErrMsg]     = useState("");

  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);
  const fileInputRef   = useRef(null);
  const activeIdRef    = useRef(activeId);
  const m = modeFor(mode, theme);

  const activeSession = sessions.find(s => s.id === activeId) || null;
  const messages      = activeSession?.messages || [];
  const isTyping      = activeId ? typingSessions.has(activeId) : false;

  useEffect(() => { activeIdRef.current = activeId; }, [activeId]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("mc_theme", theme); } catch {}
  }, [theme]);

  // Sync last exchange from active session
  useEffect(() => {
    if (!activeSession) {
      setLastUserQuery(""); setLastBotResponse(""); setLastContext(""); setLastBotMsgId(null);
      return;
    }
    const msgs = activeSession.messages;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "bot") {
        const bot = msgs[i];
        for (let j = i - 1; j >= 0; j--) {
          if (msgs[j].role === "user") {
            setLastUserQuery(msgs[j].content);
            setLastBotResponse(bot.content);
            setLastContext(bot.context_raw || ""); // ← pakai context_raw
            setLastBotMsgId(bot.id);
            return;
          }
        }
      }
    }
    setLastUserQuery(""); setLastBotResponse(""); setLastContext(""); setLastBotMsgId(null);
  }, [activeSession]);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior:"smooth" }); }, [messages, isTyping]);
  useEffect(() => { save(sessions); }, [sessions]);

  // Reset RAGAS result whenever the exchange being evaluated changes (new
  // message sent, or switched session) -- a fresh response needs a fresh eval.
  useEffect(() => {
    setRagasStatus("idle");
    setRagasMatchInfo(null);
    setRagasScores(null);
    setRagasErrMsg("");
  }, [lastUserQuery]);

  const runRagasEval = async () => {
    if (!lastUserQuery || !lastBotResponse) return;
    setRagasStatus("checking");
    setRagasMatchInfo(null);
    setRagasScores(null);
    setRagasErrMsg("");

    try {
      const matchRes = await fetch("/api/ragas/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: lastUserQuery }),
      });
      const matchData = await matchRes.json();
      if (!matchRes.ok) throw new Error(matchData.detail || "Gagal mencari ground truth");

      if (!matchData.found) {
        setRagasStatus("no_match");
        return;
      }

      setRagasMatchInfo(matchData);
      setRagasStatus("running");

      const evalRes = await fetch("/api/ragas/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query:               lastUserQuery,
          response:            lastBotResponse,
          ground_truth:        matchData.ground_truth,
          context_raw:         lastContext || "",
          mode,
          naive_kb_folder:     naiveKb.folder,
          naive_kb_filenames:  naiveKb.filenames,
        }),
      });
      const evalData = await evalRes.json();
      if (!evalRes.ok) throw new Error(evalData.detail || "Evaluasi gagal");

      setRagasScores(evalData);
      setRagasStatus("done");
    } catch (e) {
      setRagasErrMsg(e.message);
      setRagasStatus("error");
    }
  };

  /* ── Session helpers ── */
  // Pindah mode = pindah ke sesi terakhir milik mode itu (atau kosong kalau belum ada)
  const switchMode = (newMode) => {
    setMode(newMode);
    setActiveId(modeActiveId[newMode] ?? null);
    setInput("");
  };

  const newSession = () => {
    // createdMode menyimpan mode saat tombol diklik — fix bug history
    const s = { id:genId(), messages:[], ts:Date.now(), createdMode: mode };
    setSessions(p => [s, ...p]);
    setActiveId(s.id);
    setModeActiveId(prev => ({ ...prev, [mode]: s.id }));
    setInput("");
  };

  const loadSession = (s) => {
    const resolvedMode = lastMsgMode(s.messages) || s.createdMode || "naive";
    setActiveId(s.id);
    setMode(resolvedMode);
    setModeActiveId(prev => ({ ...prev, [resolvedMode]: s.id }));
    setInput("");
  };

  const deleteSession = (id, e) => {
    e.stopPropagation();
    setSessions(p => p.filter(s => s.id !== id));
    setModeActiveId(prev => {
      const next = { ...prev };
      for (const k of Object.keys(next)) if (next[k] === id) next[k] = null;
      return next;
    });
    setTypingSessions(prev => { const next = new Set(prev); next.delete(id); return next; });
    if (activeId === id) setActiveId(null);
  };

  const clearCurrent = () => {
    if (!activeId) return;
    setSessions(p => p.map(s => s.id === activeId ? { ...s, messages:[] } : s));
    setLastUserQuery(""); setLastBotResponse("");
  };

  /* ── Send ── */
  const sendMessage = async () => {
    const text = input.trim();
    if (!text || (activeId && typingSessions.has(activeId))) return;

    let sid = activeId;
    if (!sid) {
      const s = { id:genId(), messages:[], ts:Date.now(), createdMode: mode };
      setSessions(p => [s, ...p]);
      setActiveId(s.id);
      setModeActiveId(prev => ({ ...prev, [mode]: s.id }));
      sid = s.id;
      await new Promise(r => setTimeout(r, 0));
    }

    const userMsg = { role:"user", content:text, id:genId(), msgMode:mode };
    setSessions(p => p.map(s =>
      s.id === sid ? { ...s, messages:[...s.messages, userMsg], ts:Date.now() } : s
    ));
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    setTypingSessions(prev => new Set(prev).add(sid));

    try {
      const body = { query:text, mode };
      if (mode === "naive") {
        if (naiveKb.folder && naiveKb.filenames.length > 0) {
          body.kb_folder    = naiveKb.folder;
          body.kb_filenames = naiveKb.filenames;
        }
      } else {
        if (graphKb.kbPaths.length > 0) body.graph_kb_paths = graphKb.kbPaths;
      }

      const res  = await fetch("/api/chat", {
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body),
      });
      const data = await res.json();
      const content    = res.ok ? data.response : "⚠️ Error: " + (data.detail || data.response || "Terjadi kesalahan");
      const kb         = res.ok ? (data.source || data.knowledge_base || null) : null;
      // context_raw = context ASLI yang dipakai LLM (returned dari backend)
      const context_raw = res.ok ? (data.context_raw || "") : "";

      const botMsg = { role:"bot", content, id:genId(), msgMode:mode, kb, context_raw };
      setSessions(p => p.map(s =>
        s.id === sid ? { ...s, messages:[...s.messages, botMsg], ts:Date.now() } : s
      ));
      // Update last exchange untuk RAGAS — hanya kalau sesi ini masih yang sedang dilihat
      if (sid === activeIdRef.current) {
        setLastUserQuery(text);
        setLastBotResponse(content);
        setLastContext(context_raw);
      }
    } catch (err) {
      setSessions(p => p.map(s =>
        s.id === sid
          ? { ...s, messages:[...s.messages, { role:"bot", content:"⚠️ Gagal terhubung: "+err.message, id:genId(), msgMode:mode }], ts:Date.now() }
          : s
      ));
    } finally {
      setTypingSessions(prev => { const next = new Set(prev); next.delete(sid); return next; });
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  /* ── Upload ── */
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) { alert("Hanya file PDF"); return; }
    setNaiveKb(p => ({ ...p, busy:true }));
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res  = await fetch("/api/upload", { method:"POST", body:fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Gagal upload");
      setNaiveKb({ label:file.name, status:`Indexed ✓ · ${data.chunks_count} chunks`, busy:false, folder:null, filenames:[] });
    } catch (err) {
      alert("Upload gagal: " + err.message);
      setNaiveKb(p => ({ ...p, busy:false }));
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  /* ── Picker confirm ── */
  const handlePickerConfirm = async (folder, selected) => {
    setShowPicker(false);
    if (mode === "naive") {
      setNaiveKb(p => ({ ...p, busy:true, label:"", status:"" }));
      try {
        const res  = await fetch("/api/naive/index", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body:JSON.stringify({ folder, filenames:selected }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Gagal mengindeks");
        const lbl = selected.length === 1 ? selected[0] : `${selected.length} PDF · ${folder}`;
        setNaiveKb({ label:lbl, status:`Indexed ✓ · ${data.chunks_count} chunks · ${folder}`, busy:false, folder, filenames:selected });
      } catch (err) {
        alert("Indeks gagal: " + err.message);
        setNaiveKb(p => ({ ...p, busy:false }));
      }
    } else {
      setGraphKb(p => ({ ...p, busy:true, label:"", status:"" }));
      try {
        const res  = await fetch("/api/graph/select", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body:JSON.stringify({ folder, kb_names:selected }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Gagal memilih KB");
        const lbl = selected.length === 1 ? selected[0] : `${selected.length} KB · ${folder}`;
        setGraphKb({ label:lbl, status:`Siap · ${folder}`, busy:false, folder, kbNames:selected, kbPaths:data.kb_paths||[] });
      } catch (err) {
        alert("Pemilihan KB gagal: " + err.message);
        setGraphKb(p => ({ ...p, busy:false }));
      }
    }
  };

  const hasMixedModes = (() => {
    const modes = new Set(messages.filter(msg => msg.msgMode).map(msg => msg.msgMode));
    return modes.size > 1;
  })();

  const activeKbLabel = mode === "naive" ? naiveKb.label : graphKb.label;

  return (
    <div className="layout">

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-row">
            <span className="logo-text">Maritim<span style={{ color:m.color }}>Chat</span></span>
            <button
              className="theme-toggle-btn"
              onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
              title={theme === "dark" ? "Mode terang" : "Mode gelap"}
            >
              {theme === "dark" ? <Sun size={13}/> : <Moon size={13}/>}
            </button>
          </div>
          <span className="logo-sub">RAG Legal QA</span>
        </div>

        <button className="new-chat-btn" onClick={newSession}>
          <Plus size={14}/> Chat Baru
        </button>

        {/* Mode selector */}
        <nav className="sidebar-nav">
          <p className="nav-label">Mode Retrieval</p>
          {Object.values(MODES).map(modeInfo => {
            const active = mode === modeInfo.key;
            const mod = { ...modeInfo, color: accentFor(modeInfo.key, theme) };
            return (
              <button key={mod.key}
                className={`nav-item ${active ? "nav-active" : ""}`}
                style={active ? { borderColor:mod.color+"55", color:mod.color } : {}}
                onClick={() => switchMode(mod.key)}
              >
                <span className="nav-icon" style={active ? { color:mod.color } : {}}>{mod.icon}</span>
                <div className="nav-info">
                  <span className="nav-name">{mod.label}</span>
                  <span className="nav-desc">{mod.tagline}</span>
                </div>
                {active && <span className="nav-dot" style={{ background:mod.color }}/>}
              </button>
            );
          })}
        </nav>

        {/* KB Panel */}
        <KBPanel
          mode={mode}
          theme={theme}
          kbState={mode === "naive" ? naiveKb : graphKb}
          onUpload={handleUpload}
          onOpenPicker={() => setShowPicker(true)}
          onClear={() => {
            if (mode === "naive") {
              setNaiveKb({ label:"", status:"", busy:false, folder:null, filenames:[] });
              if (fileInputRef.current) fileInputRef.current.value = "";
            } else {
              setGraphKb({ label:"", status:"", busy:false, folder:null, kbNames:[], kbPaths:[] });
            }
          }}
          fileInputRef={fileInputRef}
        />

        {/* History */}
        <div className="sidebar-section sidebar-history">
          <button className="history-toggle history-toggle-main" onClick={() => setHistoryOpen(o => !o)}>
            <Clock size={11}/>
            <span className="nav-label" style={{ margin:0 }}>Riwayat Chat</span>
            {historyOpen ? <ChevronDown size={11}/> : <ChevronRight size={11}/>}
          </button>
          {historyOpen && (
            <div className="history-groups">
              {sessions.length === 0 && (
                <p className="picker-hint" style={{ padding:"6px 4px" }}>Belum ada percakapan</p>
              )}
              {["naive","graph"].map(mk => (
                <HistorySection key={mk} modeKey={mk} sessions={sessions} theme={theme}
                  activeId={activeId} onLoad={loadSession} onDelete={deleteSession}/>
              ))}
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <span className="footer-text">Muhamad Arjun D · Skripsi 2026</span>
        </div>
      </aside>

      {/* ── Chat Main ── */}
      <main className="chat-main">
        <header className="chat-header">
          <div className="header-info">
            <div className="header-dot" style={{ background:m.color }}/>
            <span className="header-mode" style={{ color:m.color }}>{m.label}</span>
            <span className="header-sep">·</span>
            <span className="header-sub">{activeKbLabel || m.tagline}</span>
            {hasMixedModes && (
              <span className="mixed-badge">
                <span style={{ color:accentFor("naive", theme) }}>N</span>+<span style={{ color:accentFor("graph", theme) }}>G</span>
              </span>
            )}
          </div>
          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearCurrent}>
              <Trash2 size={13}/><span>Bersihkan</span>
            </button>
          )}
        </header>

        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon" style={{ borderColor:m.color+"55", color:m.color }}>{m.icon}</div>
              <p className="empty-title">
                {activeId ? "Mulai percakapan" : "Buat chat baru atau pilih riwayat"}
              </p>
              <p className="empty-sub">
                {mode === "naive"
                  ? "Upload atau pilih PDF dari dataset, lalu ajukan pertanyaan hukum maritim."
                  : "Pilih Knowledge Base, atau biarkan auto-detect."}
              </p>
            </div>
          )}

          {messages.map(msg => <Message key={msg.id} msg={msg} theme={theme}/>)}

          {isTyping && (
            <div className="msg-row msg-bot">
              <div className="msg-avatar" style={{ background:m.color, color:inkOnAccent(theme) }}>{m.avatar}</div>
              <div className="msg-bubble bubble-bot" style={{ borderColor:m.color+"40", padding:"12px 16px" }}>
                <TypingDots color={m.color}/>
              </div>
            </div>
          )}

          {/* Attached to the latest response -- scrolls with the chat feed
              rather than sitting in a fixed dock near the input. Hidden
              while a new response is still typing (nothing to evaluate yet). */}
          {!isTyping && lastUserQuery && (
            <div className="ragas-dock">
              <button
                className={`ragas-toggle-btn ${showRagas ? "ragas-toggle-active" : ""}`}
                style={showRagas ? { borderColor: m.color+"66", color: m.color, background: m.color+"0D" } : {}}
                onClick={() => setShowRagas(o => !o)}
              >
                <BarChart2 size={13}/>
                <span>RAGAS Evaluasi</span>
                {showRagas ? <ChevronDown size={11}/> : <ChevronRight size={11}/>}
              </button>
              {showRagas && (
                <RagasPanel
                  lastUserQuery={lastUserQuery}
                  status={ragasStatus}
                  matchInfo={ragasMatchInfo}
                  scores={ragasScores}
                  errMsg={ragasErrMsg}
                  onRunEval={runRagasEval}
                  mode={mode}
                  theme={theme}
                />
              )}
            </div>
          )}

          <div ref={messagesEndRef}/>
        </div>

        <div className="chat-input-area">
          <div className="input-wrapper" style={{ "--accent":m.color }}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder={
                mode === "naive"
                  ? (activeKbLabel ? `[NaiveRAG] Tanya tentang ${activeKbLabel}…` : "[NaiveRAG] Muat KB terlebih dahulu…")
                  : (activeKbLabel ? `[GraphRAG: ${activeKbLabel}] Ajukan pertanyaan…` : "[GraphRAG] Auto-detect KB dari query…")
              }
              rows={1}
              className="chat-textarea"
              onInput={e => {
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
              }}
            />
            <button className="send-btn" onClick={sendMessage}
              disabled={!input.trim() || isTyping}
              style={input.trim() && !isTyping ? { background:m.color } : {}}
            >
              <Send size={15} strokeWidth={2.5} color={input.trim() && !isTyping ? inkOnAccent(theme) : undefined}/>
            </button>
          </div>
          <div className="input-footer">
            <span className="input-mode-tag" style={{ color:m.color }}>● {m.label}</span>
            <span className="input-hint">Enter kirim · Shift+Enter baris baru</span>
          </div>
        </div>
      </main>

      {showPicker && (
        <PickerModal mode={mode} theme={theme} onConfirm={handlePickerConfirm} onClose={() => setShowPicker(false)}/>
      )}
    </div>
  );
}