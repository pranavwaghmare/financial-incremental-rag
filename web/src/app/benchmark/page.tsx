"use client";

import { useEffect, useState } from 'react';
import { CheckCircle2, Database, Gauge, Target, Timer, Trophy } from 'lucide-react';

interface MetricSet {
  hit_at_5: number;
  page_hit_rate: number;
  mrr: number;
  accuracy: number;
}

interface BenchmarkGroup {
  mode_a_incremental: MetricSet;
  mode_b_full_rebuild: MetricSet;
  delta: {
    hit_at_5_delta: number;
    page_hit_rate_delta: number;
    mrr_delta: number;
    accuracy_delta: number;
  };
}

interface BenchmarkSummary {
  dataset: string;
  timestamp: string;
  corpus: {
    total_queries: number;
    rounds: number;
  };
  efficiency: {
    speedup_factor: number;
    time_saved_percentage: number;
    incremental_update_embedding_calls: number;
    full_rebuild_embedding_calls: number;
    embedding_calls_saved: number;
    embedding_calls_saved_percentage: number;
    chunk_reembedding_ratio: number;
  };
  quality_parity: {
    base_queries: BenchmarkGroup;
    new_queries: BenchmarkGroup;
    overall: BenchmarkGroup;
  };
}

interface BenchmarkResponse {
  report_name: string;
  summary: BenchmarkSummary;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function delta(value: number) {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
}

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadBenchmark = async () => {
      try {
        const response = await fetch(`${API_URL}/benchmark/latest`, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error((await response.json()).detail || 'Benchmark report unavailable');
        }
        setData(await response.json() as BenchmarkResponse);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Benchmark report unavailable');
      } finally {
        setLoading(false);
      }
    };

    loadBenchmark();
  }, []);

  if (loading) {
    return <div className="animate-fade-in benchmark-empty">Loading benchmark results...</div>;
  }

  if (error || !data) {
    return (
      <div className="animate-fade-in">
        <div className="page-header">
          <h1 className="page-title">Incremental Benchmark</h1>
          <p className="page-subtitle">Compare incremental indexing with a full rebuild.</p>
        </div>
        <div className="benchmark-empty benchmark-error">
          <Trophy size={32} />
          <strong>{error || 'No benchmark report available.'}</strong>
          <span>Run incremental_eval.py, then refresh this page.</span>
        </div>
      </div>
    );
  }

  const { summary } = data;
  const groups = [
    ['Base queries', summary.quality_parity.base_queries],
    ['New queries', summary.quality_parity.new_queries],
    ['Overall', summary.quality_parity.overall],
  ] as const;

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div className="benchmark-title-row">
          <div>
            <h1 className="page-title">Incremental Benchmark</h1>
            <p className="page-subtitle">Efficiency gains with retrieval quality parity.</p>
          </div>
          <span className="benchmark-report-name">{data.report_name}</span>
        </div>
      </div>

      <div className="benchmark-hero">
        <div>
          <div className="benchmark-kicker"><Trophy size={17} /> Mode A vs Mode B</div>
          <h2>Update only what changed.</h2>
          <p>Incremental indexing is measured against rebuilding the complete vector database.</p>
        </div>
        <div className="benchmark-hero-stat">
          <strong>{percentage(1 - summary.efficiency.chunk_reembedding_ratio)}</strong>
          <span>of chunks avoided re-embedding</span>
        </div>
      </div>

      <div className="benchmark-grid">
        <BenchmarkCard icon={<Gauge />} label="Speedup" value={`${summary.efficiency.speedup_factor}x`} detail={`${summary.efficiency.time_saved_percentage.toFixed(1)}% time saved`} />
        <BenchmarkCard icon={<Database />} label="Embedding calls saved" value={summary.efficiency.embedding_calls_saved.toLocaleString()} detail={`${summary.efficiency.embedding_calls_saved_percentage.toFixed(1)}% fewer calls`} />
        <BenchmarkCard icon={<Timer />} label="Incremental calls" value={summary.efficiency.incremental_update_embedding_calls.toLocaleString()} detail={`Full rebuild: ${summary.efficiency.full_rebuild_embedding_calls.toLocaleString()}`} />
        <BenchmarkCard icon={<Target />} label="Queries evaluated" value={String(summary.corpus.total_queries)} detail={`${summary.corpus.rounds} incremental rounds`} />
      </div>

      <section className="benchmark-section">
        <div className="benchmark-section-heading">
          <div>
            <h2>Retrieval quality parity</h2>
            <p>Mode A should match Mode B while doing less indexing work.</p>
          </div>
          <span className="parity-badge"><CheckCircle2 size={16} /> Compared directly</span>
        </div>

        <div className="benchmark-table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Query group</th>
                <th>Metric</th>
                <th>Incremental</th>
                <th>Full rebuild</th>
                <th>Delta</th>
              </tr>
            </thead>
            <tbody>
              {groups.map(([name, group]) => (
                <tr key={name}>
                  <td><strong>{name}</strong></td>
                  <td>Hit@5</td>
                  <td>{percentage(group.mode_a_incremental.hit_at_5)}</td>
                  <td>{percentage(group.mode_b_full_rebuild.hit_at_5)}</td>
                  <td className="benchmark-delta">{delta(group.delta.hit_at_5_delta)}</td>
                </tr>
              ))}
              {groups.map(([name, group]) => (
                <tr key={`${name}-page`}>
                  <td><strong>{name}</strong></td>
                  <td>Page hit rate</td>
                  <td>{percentage(group.mode_a_incremental.page_hit_rate)}</td>
                  <td>{percentage(group.mode_b_full_rebuild.page_hit_rate)}</td>
                  <td className="benchmark-delta">{delta(group.delta.page_hit_rate_delta)}</td>
                </tr>
              ))}
              {groups.map(([name, group]) => (
                <tr key={`${name}-mrr`}>
                  <td><strong>{name}</strong></td>
                  <td>MRR</td>
                  <td>{group.mode_a_incremental.mrr.toFixed(3)}</td>
                  <td>{group.mode_b_full_rebuild.mrr.toFixed(3)}</td>
                  <td className="benchmark-delta">{delta(group.delta.mrr_delta)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function BenchmarkCard({ icon, label, value, detail }: { icon: React.ReactNode; label: string; value: string; detail: string }) {
  return (
    <div className="benchmark-card">
      <div className="benchmark-card-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}
