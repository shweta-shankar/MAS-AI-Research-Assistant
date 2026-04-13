import { useState, useEffect, useRef } from 'react'

export default function App() {
  const [view, setView] = useState('chat') // 'chat' | 'reviewer'
  
  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <img 
          src="https://raw.githubusercontent.com/goellab/digiBONE/main/assets/logo_new_REM.png" 
          alt="digiBONE" 
          className="logo" 
          onError={(e) => e.target.src='https://via.placeholder.com/50x50?text=dB'}
        />
        <button 
          className="unlock-btn" 
          title="Reviewer Login"
          onClick={() => {
            if (view === 'reviewer') {
              setView('chat')
            } else {
              const code = prompt("Enter reviewer password:")
              if (code === "admin123") {
                setView('reviewer')
              } else if (code !== null) {
                alert("Incorrect password")
              }
            }
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
          </svg>
        </button>
      </div>

      {view === 'chat' ? <ChatView /> : <ReviewerView />}
    </div>
  )
}

function ChatView() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleAsk = async (e) => {
    e?.preventDefault()
    if (!query.trim() || loading) return

    const userQ = query
    setQuery('')
    setMessages(prev => [...prev, { role: 'user', content: userQ }])
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQ })
      })
      const data = await res.json()
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: data.answer,
        sourceTier: data.source_tier,
        sources: data.sources
      }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'ai', content: "Error connecting to server." }])
    }
    setLoading(false)
  }

  return (
    <div className="main-content">
      {messages.length === 0 ? (
        <div className="hero-state">
          <h1 className="hero-title">How can I assist you?</h1>
          <form className="search-container" onSubmit={handleAsk}>
            <input 
              type="text" 
              className="search-input" 
              placeholder="Ask digiBONE anything..." 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              autoFocus
            />
            <button type="submit" className="ask-btn" disabled={loading}>
              Ask
            </button>
          </form>
        </div>
      ) : (
        <>
          <div className="chat-window">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                {msg.role === 'user' ? (
                  <div>{msg.content}</div>
                ) : (
                  <div>
                    <div dangerouslySetInnerHTML={{ __html: msg.content.replace(/\n/g, '<br/>') }} />
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="sources-row">
                        {msg.sources.map((s, i) => {
                          const cleanName = s.replace(/\[|\]/g, '').split(':')[0];
                          return <span key={i} className="source-pill" title={s}>{cleanName}</span>
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {loading && <div className="message ai" style={{ color: '#888' }}>Thinking...</div>}
            <div ref={messagesEndRef} />
          </div>

          <form className="search-container fixed-bottom" onSubmit={handleAsk}>
            <input 
              type="text" 
              className="search-input" 
              placeholder="Ask a follow-up..." 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              autoFocus
            />
            <button type="submit" className="ask-btn" disabled={loading}>
              Ask
            </button>
          </form>
        </>
      )}
    </div>
  )
}

function ReviewerView() {
  const [pending, setPending] = useState([])

  const fetchPending = async () => {
    try {
      const res = await fetch('/api/pending')
      const data = await res.json()
      setPending(data)
    } catch (err) {
      console.error(err)
    }
  }

  useEffect(() => {
    fetchPending()
  }, [])

  return (
    <div className="dashboard-view">
      <h2 className="dashboard-title">Reviewer Console</h2>
      <p style={{ color: '#666', marginBottom: '2rem' }}>Review answers generated by the system before they are committed to the Verified Cache.</p>
      
      {pending.length === 0 && <p>No pending questions to review.</p>}
      
      {pending.map(p => (
        <ReviewCard key={p.id} item={p} onComplete={fetchPending} />
      ))}
    </div>
  )
}

function ReviewCard({ item, onComplete }) {
  const [verifiedAnswer, setVerifiedAnswer] = useState('')
  const [suggestions, setSuggestions] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleAction = async (action) => {
    setSubmitting(true)
    try {
      await fetch('/api/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          id: item.id, 
          action, 
          verified_answer: verifiedAnswer || null,
          reviewer_suggestions: suggestions || null
        })
      })
      onComplete()
    } catch (err) {
      alert("Error submitting review.")
    }
    setSubmitting(false)
  }

  return (
    <div className="pending-card">
      <h3>Q: {item.question}</h3>
      <div className="ai-answer-block">
        <strong>Original AI Answer:</strong><br/>
        {item.rag_answer}
      </div>
      
      <div className="form-group">
        <label>Provide Verified Answer (Optional — completely replaces AI answer for future queries):</label>
        <textarea 
          placeholder="Type replacement answer here..." 
          value={verifiedAnswer} 
          onChange={e => setVerifiedAnswer(e.target.value)}
        />
      </div>

      <div className="form-group">
        <label>Provide Reviewer Suggestions (Optional — allows AI to dynamically re-write original answer):</label>
        <textarea 
          placeholder="e.g. Include mention of segmental limits here..." 
          value={suggestions} 
          onChange={e => setSuggestions(e.target.value)}
        />
      </div>

      <div className="action-row">
        <button className="btn-approve" onClick={() => handleAction('approve')} disabled={submitting}>
          Approve & Save
        </button>
        <button className="btn-reject" onClick={() => handleAction('reject')} disabled={submitting}>
          Reject & Delete
        </button>
      </div>
    </div>
  )
}
