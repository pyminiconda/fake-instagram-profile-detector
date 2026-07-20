import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'
import api from '../../api/client'
import toast from 'react-hot-toast'

export default function DatasetInsights() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/admin/dataset/insights')
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { toast.error(e.response?.data?.detail || 'Failed to load dataset.'); setLoading(false) })
  }, [])

  if (loading) return (
    <div>
      <div className="page-header"><h1>Dataset Insights</h1><p>Loading dataset analytics...</p></div>
      <div className="grid-2">{[1,2].map(i=><div key={i} className="skeleton" style={{height:260,borderRadius:'var(--radius-lg)'}}/>)}</div>
    </div>
  )

  if (!data) return (
    <div>
      <div className="page-header"><h1>Dataset Insights</h1></div>
      <div className="alert alert-error">Dataset not found. Make sure <code>instafake_dataset.csv</code> is in the data/ directory.</div>
    </div>
  )

  const pieData = [
    { name: 'Fake',    value: data.fake_count,    color: 'var(--danger)' },
    { name: 'Genuine', value: data.genuine_count, color: 'var(--success)' },
  ]

  const statsBarData = Object.entries(data.numeric_stats || {}).slice(0, 6).map(([col, s]) => ({
    name: col.replace('user_','').replace(/_/g,' '),
    mean: parseFloat(s.mean?.toFixed(2) || 0),
    max:  parseFloat(s.max?.toFixed(2) || 0),
  }))

  return (
    <div>
      <div className="page-header">
        <h1>Dataset Insights</h1>
        <p>Analytics from the InstaFake training dataset</p>
      </div>

      {/* Summary cards */}
      <div className="stat-grid" style={{ marginBottom:'1.5rem' }}>
        {[
          { label:'Total Records',   value: data.total_records.toLocaleString(), color:'var(--brand-primary)' },
          { label:'Fake Profiles',   value: data.fake_count.toLocaleString(),    color:'var(--danger)' },
          { label:'Genuine Profiles',value: data.genuine_count.toLocaleString(), color:'var(--success)' },
          { label:'Features',        value: data.columns?.length || '—',         color:'var(--info)' },
        ].map(({ label, value, color }) => (
          <div key={label} className="stat-card">
            <div className="stat-label">{label}</div>
            <div className="stat-value" style={{ color }}>{value}</div>
          </div>
        ))}
      </div>

      <div className="grid-2" style={{ marginBottom:'1.5rem' }}>
        {/* Class distribution pie */}
        <div className="card">
          <h2 style={{ fontSize:'1rem', fontWeight:700, marginBottom:'1rem' }}>Class Distribution</h2>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={80} dataKey="value"
                label={({ name, percent }) => `${name} ${(percent*100).toFixed(1)}%`} labelLine={false}>
                {pieData.map((e,i) => <Cell key={i} fill={e.color}/>)}
              </Pie>
              <Tooltip contentStyle={{ background:'var(--bg-card)', border:'1px solid var(--border)', borderRadius:8 }}/>
              <Legend/>
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Feature stats bar chart */}
        <div className="card">
          <h2 style={{ fontSize:'1rem', fontWeight:700, marginBottom:'1rem' }}>Feature Mean Values</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={statsBarData} margin={{ left:-20 }}>
              <XAxis dataKey="name" tick={{ fontSize:10, fill:'var(--text-muted)' }} axisLine={false} tickLine={false}/>
              <YAxis tick={{ fontSize:10, fill:'var(--text-muted)' }} axisLine={false} tickLine={false}/>
              <Tooltip contentStyle={{ background:'var(--bg-card)', border:'1px solid var(--border)', borderRadius:8, fontSize:12 }}/>
              <Bar dataKey="mean" fill="var(--brand-primary)" radius={4} name="Mean"/>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Numeric stats table */}
      <div className="card">
        <h2 style={{ fontSize:'1rem', fontWeight:700, marginBottom:'1rem' }}>Feature Statistics</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Feature</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>
            </thead>
            <tbody>
              {Object.entries(data.numeric_stats || {}).map(([col, s]) => (
                <tr key={col}>
                  <td style={{ fontWeight:600, fontSize:'0.85rem' }}>{col}</td>
                  <td style={{ fontSize:'0.85rem' }}>{s.mean}</td>
                  <td style={{ fontSize:'0.85rem' }}>{s.std}</td>
                  <td style={{ fontSize:'0.85rem' }}>{s.min}</td>
                  <td style={{ fontSize:'0.85rem' }}>{s.max}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
