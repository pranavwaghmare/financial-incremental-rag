"use client";

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, BookOpen } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

interface Source {
  document: string;
  page: number | string;
  distance: number;
  text: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState<number | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userMessage }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response');
      }

      const data = await response.json();
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: data.answer, 
          sources: data.sources 
        }
      ]);
    } catch {
      setMessages(prev => [
        ...prev, 
        { role: 'assistant', content: 'Sorry, I encountered an error while processing your request.' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 6rem)' }}>
      <div className="page-header" style={{ marginBottom: '1rem' }}>
        <h1 className="page-title">💬 Financial AI Assistant</h1>
        <p className="page-subtitle">Ask questions about your indexed financial reports</p>
      </div>

      <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: 0 }}>
        
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {messages.length === 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
              <Bot size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
              <h3>How can I help you today?</h3>
              <p>Ask about revenue, profit, risks, or any financial metrics.</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} style={{ 
                display: 'flex', 
                gap: '1rem', 
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '80%'
              }}>
                {msg.role === 'assistant' && (
                  <div style={{ width: '36px', height: '36px', borderRadius: '50%', backgroundColor: '#ecfdf5', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <Bot size={20} color="#059669" />
                  </div>
                )}
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{
                    backgroundColor: msg.role === 'user' ? 'var(--primary)' : '#f1f5f9',
                    color: msg.role === 'user' ? 'white' : 'var(--text-main)',
                    padding: '1rem 1.25rem',
                    borderRadius: '1rem',
                    borderTopRightRadius: msg.role === 'user' ? 0 : '1rem',
                    borderTopLeftRadius: msg.role === 'assistant' ? 0 : '1rem',
                    boxShadow: 'var(--shadow-sm)',
                    lineHeight: 1.6
                  }}>
                    <div className="chat-markdown">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>

                  {msg.sources && msg.sources.length > 0 && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <button 
                        onClick={() => setExpandedSources(expandedSources === idx ? null : idx)}
                        style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '0.5rem', 
                          color: 'var(--text-muted)', 
                          fontSize: '0.875rem',
                          fontWeight: 500
                        }}
                      >
                        <BookOpen size={16} />
                        {expandedSources === idx ? 'Hide Evidence' : 'View Retrieved Evidence'}
                      </button>
                      
                      {expandedSources === idx && (
                        <div style={{ 
                          marginTop: '0.75rem', 
                          padding: '1rem', 
                          backgroundColor: '#f8fafc', 
                          border: '1px solid var(--border-color)', 
                          borderRadius: 'var(--radius-md)',
                          fontSize: '0.875rem'
                        }}>
                          {msg.sources.map((source, sIdx) => (
                            <div key={sIdx} style={{ marginBottom: sIdx !== msg.sources!.length - 1 ? '1rem' : 0, paddingBottom: sIdx !== msg.sources!.length - 1 ? '1rem' : 0, borderBottom: sIdx !== msg.sources!.length - 1 ? '1px solid var(--border-color)' : 'none' }}>
                              <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>{source.document}</div>
                              <div style={{ color: 'var(--text-muted)', marginBottom: '0.5rem', display: 'flex', gap: '1rem' }}>
                                <span>Page: {source.page}</span>
                                <span>Similarity: {source.distance.toFixed(4)}</span>
                              </div>
                              <div style={{ color: 'var(--text-main)', fontStyle: 'italic', backgroundColor: '#ffffff', padding: '0.75rem', borderRadius: '0.25rem' }}>
                                &quot;{source.text.substring(0, 400)}...&quot;
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {msg.role === 'user' && (
                  <div style={{ width: '36px', height: '36px', borderRadius: '50%', backgroundColor: '#dbeafe', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <User size={20} color="var(--primary)" />
                  </div>
                )}
              </div>
            ))
          )}
          {isLoading && (
            <div style={{ display: 'flex', gap: '1rem', alignSelf: 'flex-start' }}>
               <div style={{ width: '36px', height: '36px', borderRadius: '50%', backgroundColor: '#ecfdf5', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Bot size={20} color="#059669" />
                </div>
                <div style={{
                    backgroundColor: '#f1f5f9',
                    padding: '1rem 1.25rem',
                    borderRadius: '1rem',
                    borderTopLeftRadius: 0,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem'
                  }}>
                    <span style={{ display: 'inline-block', width: '8px', height: '8px', backgroundColor: 'var(--text-muted)', borderRadius: '50%', animation: 'fadeIn 1s infinite alternate' }}></span>
                    <span style={{ display: 'inline-block', width: '8px', height: '8px', backgroundColor: 'var(--text-muted)', borderRadius: '50%', animation: 'fadeIn 1s infinite alternate 0.3s' }}></span>
                    <span style={{ display: 'inline-block', width: '8px', height: '8px', backgroundColor: 'var(--text-muted)', borderRadius: '50%', animation: 'fadeIn 1s infinite alternate 0.6s' }}></span>
                </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div style={{ padding: '1rem 1.5rem', borderTop: '1px solid var(--border-color)', backgroundColor: '#f8fafc' }}>
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.75rem' }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about revenue, profit, risks..."
              disabled={isLoading}
              style={{
                flex: 1,
                padding: '0.875rem 1.25rem',
                borderRadius: '9999px',
                border: '1px solid var(--border-color)',
                outline: 'none',
                fontSize: '1rem',
                boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.05)'
              }}
            />
            <button 
              type="submit" 
              disabled={!input.trim() || isLoading}
              style={{
                width: '48px',
                height: '48px',
                borderRadius: '50%',
                backgroundColor: !input.trim() || isLoading ? '#94a3b8' : 'var(--primary)',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'background-color 0.2s'
              }}
            >
              <Send size={20} style={{ marginLeft: '2px' }} />
            </button>
          </form>
        </div>

      </div>
    </div>
  );
}
