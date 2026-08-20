import { useState, useCallback, useRef, useEffect } from 'react'

const API_BASE = ''

// ─── API Service ─────────────────────────────────────────
const api = {
  async search(params) {
    const res = await fetch(`${API_BASE}/api/v1/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
    return res.json()
  },

  async imageSearch(formData) {
    const res = await fetch(`${API_BASE}/api/v1/search/image`, {
      method: 'POST',
      body: formData,
    })
    return res.json()
  },

  async temporalSearch(params) {
    const res = await fetch(`${API_BASE}/api/v1/search/temporal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
    return res.json()
  },

  async getNearbyFrames(id, range = 40) {
    const res = await fetch(`${API_BASE}/api/v1/frames/nearby?id=${id}&range=${range}`)
    return res.json()
  },

  async chat(question) {
    const res = await fetch(`${API_BASE}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
    return res.json()
  },

  async automaticQuery(query) {
    const res = await fetch(`${API_BASE}/api/v1/automatic/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_text: query, max_results: 100 }),
    })
    return res.json()
  },
}

// ─── App Component ───────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState('text')
  const [queries, setQueries] = useState([])
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [searchInfo, setSearchInfo] = useState(null)
  const [selectedFrame, setSelectedFrame] = useState(null)
  const [nearbyFrames, setNearbyFrames] = useState([])
  const [chatOpen, setChatOpen] = useState(false)
  const [chatMessages, setChatMessages] = useState([])
  const [searchMode, setSearchMode] = useState('hybrid')
  const [resultLimit, setResultLimit] = useState(200)
  const [useLLM, setUseLLM] = useState(true)
  const [temporalEvents, setTemporalEvents] = useState(['', ''])
  const textInputRef = useRef(null)
  const chatInputRef = useRef(null)

  // ─── Keyboard Shortcuts ─────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault()
        handleSearch()
      }
      if (e.key === 'Escape') {
        setSelectedFrame(null)
      }
      if (e.ctrlKey && e.key === '/') {
        e.preventDefault()
        textInputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })

  // ─── Add Query ──────────────────────────────────────
  const addQuery = useCallback((text) => {
    if (!text.trim()) return
    setQueries(q => [...q, { type: 'text', value: text.trim() }])
  }, [])

  const removeQuery = useCallback((idx) => {
    setQueries(q => q.filter((_, i) => i !== idx))
  }, [])

  // ─── Search ─────────────────────────────────────────
  const handleSearch = useCallback(async () => {
    if (activeTab === 'temporal') {
      handleTemporalSearch()
      return
    }

    const textQueries = queries.filter(q => q.type === 'text').map(q => q.value)
    const imageQuery = queries.find(q => q.type === 'image')

    if (textQueries.length === 0 && !imageQuery) return

    setLoading(true)
    try {
      if (imageQuery && imageQuery.file) {
        // Image Search
        const formData = new FormData()
        formData.append('image', imageQuery.file)
        formData.append('limit', String(resultLimit))
        const data = await api.imageSearch(formData)
        setResults(data.results || [])
        setSearchInfo({ total: data.total, time: data.search_time_ms })
      } else {
        // Text Search
        const modeMap = {
          text: 'hybrid',
          ocr: 'ocr',
          object: 'object',
          place: 'place',
          audio: 'audio',
        }
        const data = await api.search({
          text_queries: textQueries,
          mode: modeMap[activeTab] || searchMode,
          limit: resultLimit,
          use_llm_enhance: useLLM,
        })
        setResults(data.results || [])
        setSearchInfo({
          total: data.total,
          time: data.search_time_ms,
          enhanced: data.query_enhanced,
        })
      }
    } catch (err) {
      console.error('Search error:', err)
    } finally {
      setLoading(false)
    }
  }, [queries, activeTab, searchMode, resultLimit, useLLM])

  // ─── Temporal Search ────────────────────────────────
  const handleTemporalSearch = useCallback(async () => {
    const events = temporalEvents.filter(e => e.trim())
    if (events.length < 2) return

    setLoading(true)
    try {
      const data = await api.temporalSearch({
        events: events.map(text => ({ text })),
        limit: 50,
        max_time_gap: 300,
      })
      // Flatten temporal matches into result grid
      const flatResults = []
      for (const match of (data.matches || [])) {
        for (const frame of match.frames) {
          flatResults.push({
            ...frame,
            _temporal_video: match.video_id,
            _temporal_score: match.total_score,
          })
        }
      }
      setResults(flatResults)
      setSearchInfo({
        total: data.total,
        time: data.search_time_ms,
        enhanced: null,
      })
    } catch (err) {
      console.error('Temporal search error:', err)
    } finally {
      setLoading(false)
    }
  }, [temporalEvents])

  // ─── Frame Explorer ─────────────────────────────────
  const openFrame = useCallback(async (result) => {
    setSelectedFrame(result)
    try {
      const data = await api.getNearbyFrames(result.id)
      setNearbyFrames(data.results || [])
    } catch (err) {
      console.error('Nearby frames error:', err)
    }
  }, [])

  // ─── Image Search ───────────────────────────────────
  const handleImageUpload = useCallback(async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setQueries(q => [...q, { type: 'image', value: file.name, file }])
  }, [])

  // ─── Chat ───────────────────────────────────────────
  const sendChat = useCallback(async (msg) => {
    if (!msg.trim()) return
    setChatMessages(m => [...m, { role: 'user', text: msg }])

    try {
      const data = await api.chat(msg)
      setChatMessages(m => [...m, { role: 'bot', text: data.text }])
    } catch {
      setChatMessages(m => [...m, { role: 'bot', text: 'Có lỗi xảy ra.' }])
    }
  }, [])

  const handleAutoSearch = useCallback(async (msg) => {
    if (!msg.trim()) return
    setChatMessages(m => [...m, { role: 'user', text: msg }])
    setChatMessages(m => [...m, { role: 'bot', text: 'Đang tự động phân tích và tìm kiếm...' }])
    
    setLoading(true)
    try {
      const data = await api.automaticQuery(msg)
      setResults(data.results || [])
      setSearchInfo({ total: data.results?.length || 0 })
      setChatMessages(m => m.slice(0, -1).concat([{ role: 'bot', text: `Đã tìm thấy ${data.results?.length || 0} kết quả! Hình ảnh đã được hiển thị trên màn hình.` }]))
    } catch {
      setChatMessages(m => m.slice(0, -1).concat([{ role: 'bot', text: 'Có lỗi xảy ra khi tìm kiếm tự động.' }]))
    } finally {
      setLoading(false)
    }
  }, [])

  // ─── Handle form submit ─────────────────────────────
  const handleFormSubmit = (e) => {
    e.preventDefault()
    const input = textInputRef.current
    if (input && input.value.trim()) {
      addQuery(input.value)
      input.value = ''
    }
  }

  const handleSearchAll = () => {
    handleSearch()
  }

  // ─── Thumbnail URL ─────────────────────────────────
  const getThumbUrl = (id) => {
    if (!id) return ''
    const parts = id.split('_')
    if (parts.length < 3) return ''
    const group = parts[0]
    const videoId = `${parts[0]}_${parts[1]}`
    const frameNum = parts[2]
    return `${API_BASE}/static/frames/${group}/${videoId}/${frameNum}.jpg`
  }

  // ─── Render ─────────────────────────────────────────
  return (
    <>
      {/* Navbar */}
      <nav className="navbar">
        <div className="navbar-brand">
          <i className="fa-solid fa-brain"></i>
          AIC 2026 — Smart Retrieval
        </div>
        <div className="navbar-status">
          <span><span className="status-dot"></span>System Ready</span>
          <span>{results.length} kết quả</span>
          <kbd>Ctrl+Enter</kbd> Search
          <kbd>Ctrl+/</kbd> Focus
        </div>
      </nav>

      <div className="app-layout">
        {/* ─── Sidebar ─────────────────────────────────── */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <div className="search-tabs">
              {[
                { id: 'text', icon: 'fa-font', label: 'Text' },
                { id: 'ocr', icon: 'fa-eye', label: 'OCR' },
                { id: 'object', icon: 'fa-cube', label: 'Object' },
                { id: 'place', icon: 'fa-location-dot', label: 'Place' },
                { id: 'audio', icon: 'fa-microphone', label: 'Audio' },
                { id: 'temporal', icon: 'fa-timeline', label: 'Temporal' },
              ].map(tab => (
                <button
                  key={tab.id}
                  className={`search-tab ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <i className={`fa-solid ${tab.icon}`}></i>
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-scroll">
            {/* Search Input */}
            <div className="sidebar-section">
              <h3><i className="fa-solid fa-magnifying-glass"></i> {activeTab === 'temporal' ? 'Chuỗi sự kiện' : 'Truy vấn'}</h3>

              {activeTab === 'temporal' ? (
                /* Temporal Search UI */
                <div className="temporal-events">
                  {temporalEvents.map((evt, i) => (
                    <div key={i}>
                      {i > 0 && <div className="temporal-connector"></div>}
                      <div className="temporal-event">
                        <div className="temporal-event-num">{i + 1}</div>
                        <input
                          placeholder={`Sự kiện ${i + 1}...`}
                          value={evt}
                          onChange={e => {
                            const newEvents = [...temporalEvents]
                            newEvents[i] = e.target.value
                            setTemporalEvents(newEvents)
                          }}
                        />
                        {temporalEvents.length > 2 && (
                          <button className="remove-event" onClick={() => {
                            setTemporalEvents(temporalEvents.filter((_, j) => j !== i))
                          }}>
                            <i className="fa-solid fa-xmark"></i>
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                  {temporalEvents.length < 4 && (
                    <button className="btn btn-secondary btn-sm" onClick={() => setTemporalEvents([...temporalEvents, ''])}>
                      <i className="fa-solid fa-plus"></i> Thêm sự kiện
                    </button>
                  )}
                </div>
              ) : (
                /* Standard Search */
                <div className="search-input-group">
                  <form onSubmit={handleFormSubmit}>
                    <textarea
                      ref={textInputRef}
                      className="search-textarea"
                      placeholder={
                        activeTab === 'ocr' ? 'Nhập text xuất hiện trong hình...' :
                        activeTab === 'object' ? 'Nhập đối tượng cần tìm...' :
                        activeTab === 'place' ? 'Nhập địa điểm...' :
                        activeTab === 'audio' ? 'Nhập nội dung audio...' :
                        'Mô tả cảnh cần tìm...'
                      }
                      onKeyDown={e => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          handleFormSubmit(e)
                        }
                      }}
                    />
                    <button type="submit" className="btn btn-secondary btn-sm btn-block">
                      <i className="fa-solid fa-plus"></i> Thêm truy vấn
                    </button>
                  </form>

                  {/* Image Upload */}
                  {activeTab === 'text' && (
                    <label className="btn btn-secondary btn-sm btn-block" style={{ cursor: 'pointer' }}>
                      <i className="fa-solid fa-image"></i> Tải ảnh lên
                      <input type="file" accept="image/*" hidden onChange={handleImageUpload} />
                    </label>
                  )}
                </div>
              )}
            </div>

            {/* Query Tags */}
            {queries.length > 0 && (
              <div className="sidebar-section">
                <h3>Danh sách truy vấn ({queries.length})</h3>
                <div className="query-tags">
                  {queries.map((q, i) => (
                    <div key={i} className={`query-tag ${q.type === 'image' ? 'image-tag' : ''}`}>
                      <i className={`fa-solid ${q.type === 'image' ? 'fa-image' : 'fa-quote-left'}`}></i>
                      <span className="tag-text">{q.value}</span>
                      <span className="tag-remove" onClick={() => removeQuery(i)}>
                        <i className="fa-solid fa-xmark"></i>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Search Options */}
            <div className="sidebar-section">
              <h3>Tùy chọn</h3>
              <div className="search-options">
                <div className="option-row">
                  <span className="option-label">Số kết quả</span>
                  <select className="option-select" value={resultLimit} onChange={e => setResultLimit(+e.target.value)}>
                    {[100, 200, 300, 500, 1000, 1500, 2000].map(n => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </div>
                <div className="option-row">
                  <span className="option-label">LLM Enhance</span>
                  <div className={`toggle ${useLLM ? 'active' : ''}`} onClick={() => setUseLLM(!useLLM)}></div>
                </div>
              </div>
            </div>

            {/* Search Button */}
            <div className="sidebar-section">
              <button
                className="btn btn-primary btn-block"
                onClick={handleSearchAll}
                disabled={loading || (activeTab !== 'temporal' && queries.length === 0)}
              >
                {loading ? (
                  <><div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></div> Đang tìm...</>
                ) : (
                  <><i className="fa-solid fa-magnifying-glass"></i> Tìm kiếm</>
                )}
              </button>
              {queries.length > 0 && (
                <button className="btn btn-danger btn-sm btn-block" style={{ marginTop: 8 }} onClick={() => { setQueries([]); setResults([]); setSearchInfo(null) }}>
                  <i className="fa-solid fa-trash"></i> Xóa tất cả
                </button>
              )}
            </div>
          </div>
        </aside>

        {/* ─── Main Content ────────────────────────────── */}
        <main className="main-content">
          {/* Results Header */}
          {searchInfo && (
            <div className="results-header">
              <div className="results-info">
                <span className="results-count">{searchInfo.total} kết quả</span>
                {searchInfo.time && <span className="results-time">{searchInfo.time}ms</span>}
              </div>
              {searchInfo.enhanced && (
                <div className="enhanced-queries">
                  <i className="fa-solid fa-wand-magic-sparkles"></i>
                  {searchInfo.enhanced.slice(0, 3).map((q, i) => (
                    <span key={i} className="enhanced-tag">{q}</span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Results Grid */}
          {loading ? (
            <div className="loading-spinner"><div className="spinner"></div></div>
          ) : results.length > 0 ? (
            <div className="results-grid">
              {results.map((r, i) => (
                <div
                  key={r.id}
                  className={`result-card ${selectedFrame?.id === r.id ? 'selected' : ''}`}
                  onClick={() => openFrame(r)}
                >
                  <div className="result-rank">{i + 1}</div>
                  <img
                    src={r.thumbnail_url || getThumbUrl(r.id)}
                    alt={r.id}
                    loading="lazy"
                    onError={e => { e.target.style.display = 'none' }}
                  />
                  <div className="result-card-overlay">
                    <span className="result-id">{r.id}</span>
                    <span className="result-score">{(r.score * 100).toFixed(1)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <i className="fa-solid fa-video"></i>
              <p>Thêm truy vấn và nhấn <strong>Tìm kiếm</strong> để bắt đầu</p>
              <div className="shortcut-hint">
                <kbd>Ctrl+/</kbd> Focus input
                <kbd>Ctrl+Enter</kbd> Search
                <kbd>Esc</kbd> Close explorer
              </div>
            </div>
          )}
        </main>

        {/* ─── Frame Explorer ──────────────────────────── */}
        <aside className={`explorer-panel ${!selectedFrame ? 'hidden' : ''}`}>
          {selectedFrame && (
            <>
              <div className="explorer-header">
                <h3><i className="fa-solid fa-image"></i> {selectedFrame.id}</h3>
                <button className="explorer-close" onClick={() => setSelectedFrame(null)}>
                  <i className="fa-solid fa-xmark"></i>
                </button>
              </div>
              <div className="explorer-main-frame">
                <img
                  src={selectedFrame.thumbnail_url || getThumbUrl(selectedFrame.id)}
                  alt={selectedFrame.id}
                />
              </div>
              <div className="explorer-meta">
                <div className="explorer-meta-row">
                  <span>Video ID</span>
                  <span>{selectedFrame.metadata?.video_id || '-'}</span>
                </div>
                <div className="explorer-meta-row">
                  <span>Frame</span>
                  <span>{selectedFrame.metadata?.n || '-'}</span>
                </div>
                <div className="explorer-meta-row">
                  <span>Time</span>
                  <span>{selectedFrame.metadata?.pts_time?.toFixed(2) || '-'}s</span>
                </div>
                <div className="explorer-meta-row">
                  <span>Score</span>
                  <span>{(selectedFrame.score * 100).toFixed(2)}</span>
                </div>
                {selectedFrame.metadata?.youtube_id && (
                  <a
                    href={`https://www.youtube.com/watch?v=${selectedFrame.metadata.youtube_id}&t=${Math.floor(selectedFrame.metadata.pts_time || 0)}`}
                    target="_blank"
                    rel="noreferrer"
                    className="btn btn-secondary btn-sm"
                    style={{ marginTop: 8 }}
                  >
                    <i className="fa-brands fa-youtube"></i> Xem trên YouTube
                  </a>
                )}
              </div>
              <div className="explorer-nearby-title">Frames lân cận</div>
              <div className="explorer-nearby-grid">
                {nearbyFrames.map(f => (
                  <div
                    key={f.id}
                    className={`result-card ${f.id === selectedFrame.id ? 'selected' : ''}`}
                    onClick={() => openFrame({ ...f, score: 0, thumbnail_url: f.thumbnail_url })}
                  >
                    <img
                      src={f.thumbnail_url || getThumbUrl(f.id)}
                      alt={f.id}
                      loading="lazy"
                      onError={e => { e.target.style.display = 'none' }}
                    />
                    <div className="result-card-overlay">
                      <span className="result-id">{f.id.split('_').pop()}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </aside>
      </div>

      {/* ─── Chat Widget ─────────────────────────────── */}
      <div className="chat-widget">
        <div className={`chat-panel ${!chatOpen ? 'hidden' : ''}`}>
          <div className="chat-header">
            <i className="fa-solid fa-robot"></i> Trợ lý AI
          </div>
          <div className="chat-messages">
            {chatMessages.length === 0 && (
              <div className="chat-msg bot">Xin chào! Tôi có thể giúp bạn tìm kiếm video. Hãy mô tả cảnh bạn muốn tìm.</div>
            )}
            {chatMessages.map((m, i) => (
              <div key={i} className={`chat-msg ${m.role}`}>{m.text}</div>
            ))}
          </div>
          <div className="chat-input-area">
            <input
              ref={chatInputRef}
              className="chat-input"
              placeholder="Nhập câu hỏi..."
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  sendChat(e.target.value)
                  e.target.value = ''
                }
              }}
            />
            <button className="chat-send" title="Trò chuyện" onClick={() => {
              if (chatInputRef.current) {
                sendChat(chatInputRef.current.value)
                chatInputRef.current.value = ''
              }
            }}>
              <i className="fa-solid fa-paper-plane"></i>
            </button>
            <button className="chat-auto" title="Tìm tự động" onClick={() => {
              if (chatInputRef.current) {
                handleAutoSearch(chatInputRef.current.value)
                chatInputRef.current.value = ''
              }
            }} style={{ background: 'var(--primary)', color: 'white', border: 'none', borderRadius: 8, padding: '0 12px', cursor: 'pointer', marginLeft: 6 }}>
              <i className="fa-solid fa-wand-magic-sparkles"></i>
            </button>
          </div>
        </div>
        <button className="chat-toggle" onClick={() => setChatOpen(!chatOpen)}>
          <i className={`fa-solid ${chatOpen ? 'fa-xmark' : 'fa-comments'}`}></i>
        </button>
      </div>
    </>
  )
}
