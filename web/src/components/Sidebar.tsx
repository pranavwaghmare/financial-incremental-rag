"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FileUp, MessageSquare, BarChart3, Database, Trophy } from 'lucide-react';

export default function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { name: 'Upload Report', path: '/upload', icon: <FileUp size={20} /> },
    { name: 'Financial Assistant', path: '/chat', icon: <MessageSquare size={20} /> },
    { name: 'Analytics Dashboard', path: '/analytics', icon: <BarChart3 size={20} /> },
    { name: 'Benchmark Results', path: '/benchmark', icon: <Trophy size={20} /> },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Database className="sidebar-logo" color="var(--primary)" size={28} />
        <span className="sidebar-title">FinDoc AI</span>
      </div>
      <div className="sidebar-subtitle">
        Incremental RAG System
      </div>

      <nav className="nav-links">
        {navItems.map((item) => {
          const isActive = pathname === item.path || (pathname === '/' && item.path === '/upload');
          return (
            <Link 
              key={item.path} 
              href={item.path}
              className={`nav-item ${isActive ? 'active' : ''}`}
            >
              {item.icon}
              {item.name}
            </Link>
          );
        })}
      </nav>

      <div style={{ marginTop: 'auto' }}>
        <div style={{ 
          padding: '1rem', 
          backgroundColor: '#ecfdf5', 
          color: '#047857', 
          borderRadius: 'var(--radius-md)',
          fontSize: '0.875rem',
          fontWeight: 500,
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <span style={{ 
            width: '8px', 
            height: '8px', 
            backgroundColor: '#10b981', 
            borderRadius: '50%',
            display: 'inline-block' 
          }}></span>
          Backend Connected
        </div>
      </div>
    </aside>
  );
}
