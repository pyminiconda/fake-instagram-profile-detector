import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Play, CheckCircle, AlertTriangle, BarChart2 } from 'lucide-react'
import api from '../../api/client'
import toast from 'react-hot-toast'

const ALGORITHMS = ['RandomForest', 'XGBoost', 'LogisticRegression', 'GradientBoosting', 'SVM']

const ALGO_DESC = {
  RandomForest: 'Ensemble of decision trees. Robust, fast, excellent for tabular data.',
  XGBoost: 'Gradient boosting with regularization. Often top performer on structured data.',
  LogisticRegression: 'Simple linear classifier. Good baseline, highly interpretable.',
  GradientBoosting: 'Sklearn gradient boosting. Slower but often very accurate.',
  SVM: 'Support Vector Machine. Effective for high-dimensional spaces and structured classification.',
}

export default function ModelTraining() {
  const [algo, setAlgo]         = useState('RandomForest')
  const [training, setTraining] = useState(false)
  const [status, setStatus]     = useState(null)
  const [loading, setLoading]   = useState(true)
  const [reloading, setReloading] = useState(false)

  const reloadModel = async () => {
    setReloading(true)
    try {
      await api.post('/admin/model/reload')
      toast.success('Inference engine reloaded with the latest model!')
      loadStatus()
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Failed to reload model.')
    } finally {
      setReloading(false)
    }
  }

  const loadStatus = () => {
    setLoading(true)
    api.get('/admin/model/status')
      .then(r => { setStatus(r.data); setLoading(false) })
      .catch(() => setLoading(false))
  }
  useEffect(() => { loadStatus() }, [])

  const startTraining = async () => {
    setTraining(true)
    try {
      await api.post(`/admin/model/train?algorithm=${algo}`)
      toast.success(`Training started for ${algo}! This may take a few minutes.`)
      // Poll for completion every 5s (up to 10 times)
      let attempts = 0
      const poll = setInterval(async () => {
        attempts++
        try {
          const r = await api.get('/admin/model/status')
          setStatus(r.data)
          if (r.data.model_ready || attempts >= 10) {
            clearInterval(poll); setTraining(false)
            if (r.data.model_ready) toast.success('Model training complete!')
          }
        } catch { clearInterval(poll); setTraining(false) }
      }, 5000)
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Training failed to start.')
      setTraining(false)
    }
  }

  const best = status?.best_model
  const allModels = status?.all_models || []

  const metricColor = (v) => {
    if (!v) return 'var(--text-muted)'
    if (v >= 0.9) return 'var(--success)'
    if (v >= 0.75) return 'var(--warning)'
    return 'var(--danger)'
  }

  return (
    <div>
      <div className="page-header">
        <h1>Model Training</h1>
        <p>Train, evaluate, and manage the fake profile detection model</p>
      </div>

      {/* Current model status */}
      <div className="card" style={{ marginBottom:'1.25rem' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'0.6rem', marginBottom:'1.25rem' }}>
          <div style={{ width:36,height:36,borderRadius:'var(--radius-md)',background:'rgba(108,99,255,0.12)',display:'flex',alignItems:'center',justifyContent:'center',color:'var(--brand-secondary)' }}>
            <Brain size={18}/>
          </div>
          <h2 style={{ fontSize:'1rem', fontWeight:700 }}>Current Best Model</h2>
          <span style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:'1rem' }}>
            <button className="btn btn-primary btn-sm" onClick={reloadModel} disabled={reloading}>
              {reloading ? <span className="spinner"/> : 'Deploy to Production'}
            </button>
            <span style={{ display:'inline-flex', alignItems:'center', gap:'0.4rem', fontSize:'0.85rem' }}>
              <span style={{ width:8,height:8,borderRadius:'50%',background:status?.model_ready?'var(--success)':'var(--danger)',display:'inline-block' }}/>
              {status?.model_ready ? 'Online' : 'Not Loaded'}
            </span>
          </span>
        </div>

        {loading ? (
          <div style={{ display:'flex', flexDirection:'column', gap:'0.5rem' }}>
            {[1,2,3].map(i => <div key={i} className="skeleton" style={{ height:36, borderRadius:'var(--radius-sm)' }}/>)}
          </div>
        ) : best ? (
          <div className="grid-3" style={{ marginBottom:'0.5rem' }}>
            {[
              { label:'Algorithm', value: best.algorithmType },
              { label:'Version',   value: best.version || '1.0' },
              { label:'Trained',   value: best.trainedAt?.slice(0,10) },
              { label:'Accuracy',  value: best.accuracy  ? `${(best.accuracy*100).toFixed(1)}%`  : '—' },
              { label:'F1 Score',  value: best.f1Score   ? `${(best.f1Score*100).toFixed(1)}%`   : '—' },
              { label:'AUC-ROC',   value: best.aucRoc    ? `${(best.aucRoc*100).toFixed(1)}%`    : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="stat-card" style={{ padding:'0.875rem' }}>
                <div className="stat-label" style={{ marginBottom:'0.25rem' }}>{label}</div>
                <div style={{ fontWeight:700, fontSize:'1.1rem', color: label.includes('%')||label==='Accuracy'||label.includes('Score')||label.includes('AUC') ? metricColor(best[label]) : 'var(--text-primary)' }}>{value}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="alert alert-warning">
            <AlertTriangle size={18}/> No model trained yet. Train one below.
          </div>
        )}
      </div>

      {/* Train new model */}
      <div className="card" style={{ marginBottom:'1.25rem' }}>
        <h2 style={{ fontSize:'1rem', fontWeight:700, marginBottom:'1.25rem' }}>Train New Model</h2>

        <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:'0.75rem', marginBottom:'1.25rem' }}>
          {ALGORITHMS.map(a => (
            <button key={a} onClick={() => setAlgo(a)}
              style={{
                padding:'0.875rem 1rem', borderRadius:'var(--radius-md)', textAlign:'left', cursor:'pointer',
                background: algo===a ? 'rgba(108,99,255,0.12)' : 'var(--bg-elevated)',
                border: `1.5px solid ${algo===a ? 'var(--brand-primary)' : 'var(--border)'}`,
                transition:'all var(--transition)',
              }}>
              <div style={{ fontWeight:700, fontSize:'0.9rem', color: algo===a ? 'var(--brand-secondary)' : 'var(--text-primary)', marginBottom:'0.2rem' }}>{a}</div>
              <div style={{ fontSize:'0.75rem', color:'var(--text-muted)' }}>{ALGO_DESC[a]}</div>
            </button>
          ))}
        </div>

        <AnimatePresence>
          {training && (
            <motion.div initial={{ opacity:0, height:0 }} animate={{ opacity:1, height:'auto' }} exit={{ opacity:0, height:0 }}
              style={{ marginBottom:'1rem' }}>
              <div className="alert alert-info" style={{ display:'flex', alignItems:'center', gap:'0.75rem' }}>
                <span className="spinner"/>
                Training <strong>{algo}</strong> in the background... This may take a few minutes.
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <button className="btn btn-primary" onClick={startTraining} disabled={training} style={{ minWidth:180 }}>
          {training ? <><span className="spinner"/>Training...</> : <><Play size={16}/>Start Training</>}
        </button>
        <div style={{ fontSize:'0.8rem', color:'var(--text-muted)', marginTop:'0.6rem' }}>
            Training runs in the background. You must manually deploy the model after it finishes.
        </div>
      </div>

      {/* All models history */}
      {allModels.length > 0 && (
        <div className="card">
          <h2 style={{ fontSize:'1rem', fontWeight:700, marginBottom:'1rem' }}>Training History</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Algorithm</th><th>Accuracy</th><th>F1</th><th>AUC-ROC</th><th>Trained</th><th>Status</th></tr>
              </thead>
              <tbody>
                {allModels.map(m => (
                  <tr key={m.modelId}>
                    <td style={{ fontWeight:600 }}>{m.algorithmType}</td>
                    <td style={{ color: metricColor(m.accuracy) }}>{m.accuracy ? `${(m.accuracy*100).toFixed(1)}%` : '—'}</td>
                    <td style={{ color: metricColor(m.f1Score) }}>{m.f1Score ? `${(m.f1Score*100).toFixed(1)}%` : '—'}</td>
                    <td style={{ color: metricColor(m.aucRoc) }}>{m.aucRoc ? `${(m.aucRoc*100).toFixed(1)}%` : '—'}</td>
                    <td style={{ fontSize:'0.8rem', color:'var(--text-muted)' }}>{m.trainedAt?.slice(0,10)}</td>
                    <td>
                      {m.isBestModel
                        ? <span className="badge badge-genuine"><CheckCircle size={11}/> Best</span>
                        : <span style={{ fontSize:'0.75rem', color:'var(--text-muted)' }}>—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
