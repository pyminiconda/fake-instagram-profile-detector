import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Users, Search, ShieldCheck, BarChart2, Activity } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import api from '../../api/client'

export default function AdminDashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/admin/stats').then(r => { setStats(r.data); setLoading(false) }).catch(() => setLoading(false))
  }, [])

  if (loading) return (
    <div>
      <div className="page-header"><h1>Admin Dashboard</h1><p>System overview and statistics</p></div>
      <div className="stat-grid">{[1,2,3,4].map(i=><div key={i} className="skeleton" style={{height:100,borderRadius:'var(--radius-lg)'}}/>)}</div>
    </div>
  )

  const statCards = stats ? [
    { label:'Total Users',    value: stats.total_users,   color:'var(--brand-primary)',  icon: Users },
    { label:'Total Scans',    value: stats.total_scans,   color:'var(--info)',           icon: BarChart2 },
    { label:'Scans Today',    value: stats.scans_today,   color:'var(--success)',        icon: Activity },
    { label:'Fake Profiles',  value: stats.fake_count,    color:'var(--danger)',         icon: ShieldCheck },
    { label:'Genuine Profiles',value:stats.genuine_count, color:'var(--success)',        icon: ShieldCheck },
    { label:'Fake Rate',      value:`${stats.fake_percentage}%`, color:'var(--warning)', icon: BarChart2 },
  ] : []

  const chartData = stats ? [
    { name:'Fake',    value: stats.fake_count,    fill:'var(--danger)' },
    { name:'Genuine', value: stats.genuine_count, fill:'var(--success)' },
  ] : []

  return (
    <div>
      <div className="page-header">
        <h1>Admin Dashboard</h1>
        <p>System-wide overview and analytics</p>
      </div>

      <div className="stat-grid" style={{ marginBottom:'1.5rem' }}>
        {statCards.map(({ label, value, color, icon: Icon }, i) => (
          <motion.div key={label} className="stat-card" initial={{ opacity:0, y:16 }} animate={{ opacity:1, y:0 }} transition={{ delay: i*0.07 }}>
            <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
              <div className="stat-label">{label}</div>
              <Icon size={16} color={color} opacity={0.6}/>
            </div>
            <div className="stat-value" style={{ color }}>{value}</div>
          </motion.div>
        ))}
      </div>

      <div className="grid-2">
        {/* Chart */}
        <div className="card">
          <h2 style={{ fontSize:'1rem', fontWeight:700, marginBottom:'1rem' }}>Scan Results Breakdown</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" tick={{ fontSize:12, fill:'var(--text-secondary)' }} axisLine={false} tickLine={false}/>
              <YAxis tick={{ fontSize:11, fill:'var(--text-muted)' }} axisLine={false} tickLine={false}/>
              <Tooltip contentStyle={{ background:'var(--bg-card)', border:'1px solid var(--border)', borderRadius:8 }}/>
              <Bar dataKey="value" radius={6} fill="var(--brand-primary)"/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model status */}
        <div className="card">
          <h2 style={{ fontSize:'1rem', fontWeight:700, marginBottom:'1rem' }}>Model Status</h2>
          <div style={{ display:'flex', alignItems:'center', gap:'0.75rem', marginBottom:'1rem' }}>
            <span style={{ width:10, height:10, borderRadius:'50%', background: stats?.model_ready ? 'var(--success)' : 'var(--danger)', display:'inline-block', boxShadow: stats?.model_ready ? '0 0 8px var(--success)' : 'none' }}/>
            <span style={{ fontWeight:600 }}>{stats?.model_ready ? 'Model Online' : 'Model Not Loaded'}</span>
          </div>
          {stats?.model_algorithm && (
            <div style={{ display:'flex', flexDirection:'column', gap:'0.5rem' }}>
              {[['Algorithm', stats.model_algorithm], ['Accuracy', stats.model_accuracy ? `${(stats.model_accuracy*100).toFixed(1)}%` : 'N/A']].map(([k,v]) => (
                <div key={k} style={{ display:'flex', justifyContent:'space-between', fontSize:'0.875rem', padding:'0.4rem 0', borderBottom:'1px solid var(--border)' }}>
                  <span style={{ color:'var(--text-secondary)' }}>{k}</span>
                  <span style={{ fontWeight:600 }}>{v}</span>
                </div>
              ))}
            </div>
          )}
          <a href="/admin/model" className="btn btn-secondary btn-sm btn-full" style={{ marginTop:'1rem' }}>
            Manage Model
          </a>
        </div>
      </div>
    </div>
  )
}
