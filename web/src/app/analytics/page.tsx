"use client";

import { useState, useEffect } from 'react';
import { BarChart, Bar, PieChart, Pie, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import { Layers, Building, Database, Activity } from 'lucide-react';

export default function AnalyticsPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('indexing');

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/analytics`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
          },
        });
        if (!response.ok) {
          throw new Error('Analytics unavailable');
        }
        const jsonData = await response.json();
        setData(jsonData);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  if (loading) {
    return <div className="animate-fade-in" style={{ padding: '2rem' }}>Loading analytics...</div>;
  }

  if (error) {
    return (
      <div className="animate-fade-in">
        <div className="page-header">
          <h1 className="page-title">Financial Intelligence Dashboard</h1>
        </div>
        <div style={{ padding: '1rem', backgroundColor: '#fef2f2', color: '#b91c1c', borderRadius: 'var(--radius-md)' }}>
          {error}
        </div>
      </div>
    );
  }

  const incrementalData = [
    { name: 'New', count: data.incremental.new_chunks },
    { name: 'Modified', count: data.incremental.modified_chunks },
    { name: 'Deleted', count: data.incremental.deleted_chunks },
    { name: 'Reused', count: data.incremental.unchanged_chunks },
  ];

  const embeddingData = [
    { name: 'Embedding Calls', value: data.incremental.embedding_calls },
    { name: 'Saved Calls', value: data.incremental.embedding_calls_saved },
  ];

  const COLORS = ['#2563eb', '#10b981', '#ef4444', '#f59e0b'];
  const PIE_COLORS = ['#3b82f6', '#10b981'];

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Financial Intelligence Dashboard</h1>
        <p className="page-subtitle">Real-time metrics and system performance</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
        <MetricCard title="Reports Indexed" value={data.documents.reports} icon={<Layers />} />
        <MetricCard title="Companies" value={data.documents.companies} icon={<Building />} />
        <MetricCard title="Total Chunks" value={data.documents.chunks} icon={<Database />} />
        <MetricCard title="Queries" value={data.queries.total_queries} icon={<Activity />} />
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border-color)' }}>
          <TabButton active={activeTab === 'indexing'} onClick={() => setActiveTab('indexing')} label="Indexing" />
          <TabButton active={activeTab === 'performance'} onClick={() => setActiveTab('performance')} label="Performance" />
          <TabButton active={activeTab === 'query'} onClick={() => setActiveTab('query')} label="Query Analytics" />
        </div>

        <div style={{ padding: '2rem' }}>
          {activeTab === 'indexing' && (
            <div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1.5rem' }}>Incremental Chunk Updates</h3>
              <div style={{ height: 400, width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={incrementalData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} />
                    <YAxis axisLine={false} tickLine={false} />
                    <Tooltip cursor={{fill: '#f1f5f9'}} contentStyle={{ borderRadius: '0.5rem', border: 'none', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }} />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]} minPointSize={5}>
                      {incrementalData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {activeTab === 'performance' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
              <div>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1.5rem' }}>Embedding Optimization</h3>
                <div style={{ height: 350, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={embeddingData}
                        cx="50%"
                        cy="50%"
                        innerRadius={80}
                        outerRadius={120}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {embeddingData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ borderRadius: '0.5rem', border: 'none', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem' }}>
                  {embeddingData.map((entry, index) => (
                    <div key={entry.name} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: PIE_COLORS[index] }} />
                      <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>{entry.name}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div style={{ padding: '2rem', backgroundColor: '#f8fafc', borderRadius: 'var(--radius-lg)', textAlign: 'center' }}>
                  <div style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Avg Processing Time</div>
                  <div style={{ fontSize: '3rem', fontWeight: 700, color: 'var(--text-main)' }}>
                    {data.incremental.average_processing_time} <span style={{ fontSize: '1.5rem', color: 'var(--text-muted)' }}>sec</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'query' && (
            <div style={{ maxWidth: '600px', margin: '0 auto' }}>
              <div style={{ padding: '3rem', backgroundColor: '#f8fafc', borderRadius: 'var(--radius-lg)', textAlign: 'center' }}>
                <div style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '1.125rem' }}>Average Response Time</div>
                <div style={{ fontSize: '4rem', fontWeight: 700, color: 'var(--primary)', marginBottom: '2rem' }}>
                  {data.queries.average_response_time} <span style={{ fontSize: '1.5rem', color: 'var(--text-muted)' }}>sec</span>
                </div>
                
                <div style={{ width: '100%', height: '12px', backgroundColor: '#e2e8f0', borderRadius: '6px', overflow: 'hidden' }}>
                  <div style={{ 
                    height: '100%', 
                    backgroundColor: 'var(--primary)', 
                    width: `${Math.min((data.queries.average_response_time / 10) * 100, 100)}%`,
                    transition: 'width 1s ease-in-out'
                  }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                  <span>0s</span>
                  <span>10s+</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, icon }: { title: string, value: number, icon: React.ReactNode }) {
  return (
    <div style={{
      background: 'var(--gradient-dark)',
      color: 'white',
      padding: '1.5rem',
      borderRadius: 'var(--radius-lg)',
      boxShadow: 'var(--shadow-md)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <div style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ color: '#94a3b8', fontSize: '0.875rem', fontWeight: 500, marginBottom: '0.5rem' }}>{title}</div>
        <div style={{ fontSize: '2.5rem', fontWeight: 700 }}>{value}</div>
      </div>
      <div style={{ position: 'absolute', right: '-10px', bottom: '-10px', opacity: 0.2, transform: 'scale(2.5)' }}>
        {icon}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, label }: { active: boolean, onClick: () => void, label: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '1rem 2rem',
        fontWeight: 600,
        color: active ? 'var(--primary)' : 'var(--text-muted)',
        borderBottom: active ? '2px solid var(--primary)' : '2px solid transparent',
        backgroundColor: active ? '#f8fafc' : 'transparent',
        transition: 'all 0.2s',
      }}
    >
      {label}
    </button>
  );
}
