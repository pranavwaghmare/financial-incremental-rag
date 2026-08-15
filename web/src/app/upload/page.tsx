"use client";

import { useState } from 'react';
import { UploadCloud, File, FileText } from 'lucide-react';

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Upload failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Upload Financial Report</h1>
        <p className="page-subtitle">Upload annual reports or financial statements for indexing</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>
        <div className="card">
          <div style={{
            border: '2px dashed var(--border-color)',
            borderRadius: 'var(--radius-lg)',
            padding: '3rem',
            textAlign: 'center',
            backgroundColor: '#f8fafc',
            marginBottom: '1.5rem'
          }}>
            <input 
              type="file" 
              accept=".pdf" 
              onChange={handleFileChange}
              style={{ display: 'none' }}
              id="file-upload"
            />
            <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
              <UploadCloud size={48} color="var(--primary)" />
              <div>
                <span style={{ color: 'var(--primary)', fontWeight: 600 }}>Click to upload</span> or drag and drop
                <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.25rem' }}>PDF up to 50MB</p>
              </div>
            </label>
          </div>

          {file && (
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              padding: '1rem',
              backgroundColor: '#f1f5f9',
              borderRadius: 'var(--radius-md)',
              marginBottom: '1.5rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <File size={24} color="var(--primary)" />
                <div>
                  <div style={{ fontWeight: 500 }}>{file.name}</div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                    {(file.size / 1024).toFixed(2)} KB
                  </div>
                </div>
              </div>
            </div>
          )}

          <button 
            onClick={handleUpload}
            disabled={!file || isUploading}
            style={{
              width: '100%',
              padding: '1rem',
              backgroundColor: !file || isUploading ? '#94a3b8' : 'var(--primary)',
              color: 'white',
              borderRadius: 'var(--radius-md)',
              fontWeight: 600,
              fontSize: '1rem',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'background-color 0.2s',
              cursor: !file || isUploading ? 'not-allowed' : 'pointer'
            }}
          >
            {isUploading ? 'Extracting, chunking and indexing...' : '🚀 Process & Index'}
          </button>

          {error && (
            <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#fef2f2', color: '#b91c1c', borderRadius: 'var(--radius-md)' }}>
              {error}
            </div>
          )}

          {result && (
            <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 'var(--radius-md)' }}>
              <h3 style={{ color: '#166534', marginBottom: '0.5rem', fontWeight: 600 }}>Document indexed successfully</h3>
              <pre style={{ 
                backgroundColor: '#ffffff', 
                padding: '1rem', 
                borderRadius: '0.5rem',
                overflowX: 'auto',
                fontSize: '0.875rem',
                border: '1px solid #e2e8f0'
              }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>

        <div className="card" style={{ height: 'fit-content' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileText size={20} color="var(--primary)" />
            Supported Documents
          </h3>
          
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {['Annual Reports', 'Balance Sheets', 'Income Statements', 'Investor Presentations', 'Financial Filings'].map((doc) => (
              <li key={doc} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--text-main)' }}>
                <div style={{ width: '24px', height: '24px', borderRadius: '50%', backgroundColor: '#dbeafe', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary)', fontWeight: 'bold' }}>✓</div>
                {doc}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
