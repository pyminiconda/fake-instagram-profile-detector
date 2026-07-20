import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { History as HistoryIcon, Trash2, Download, FileText, Calendar, Search } from 'lucide-react'
import api from '../api/client'
import toast from 'react-hot-toast'

function HistorySkeleton() {
  return (
    <div style={{display:'flex',flexDirection:'column',gap:'0.5rem'}}>
      {[1,2,3,4,5].map(i => (
        <div key={i} className="skeleton" style={{height:52,borderRadius:'var(--radius-md)'}}/>
      ))}
    </div>
  )
}

export default function History() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch]   = useState('')
  const [startDate, setStart] = useState('')
  const [endDate, setEnd]     = useState('')
  const [deleting, setDeleting] = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const params = {}
      if (startDate) params.start_date = startDate
      if (endDate)   params.end_date   = endDate
      const { data } = await api.get('/history', { params })
      setRecords(data.records)
    } catch { toast.error('Failed to load history.') }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const deleteRecord = async (id) => {
    setDeleting(id)
    try {
      await api.delete(`/history/${id}`)
      setRecords(r => r.filter(x => x.historyId !== id))
      toast.success('Record deleted.')
    } catch { toast.error('Failed to delete.') }
    finally { setDeleting(null) }
  }

  const exportCSV = async () => {
    try {
      const res = await api.get('/history/export/csv', { responseType:'blob' })
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a'); a.href=url; a.download='history.csv'; a.click()
      URL.revokeObjectURL(url)
      toast.success('CSV downloaded!')
    } catch { toast.error('Export failed.') }
  }

  const exportPDF = async () => {
    try {
      const res = await api.get('/history/export/pdf', { responseType:'blob' })
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a'); a.href=url; a.download='history_report.pdf'; a.click()
      URL.revokeObjectURL(url)
      toast.success('PDF downloaded!')
    } catch { toast.error('Export failed.') }
  }

  const filtered = records.filter(r =>
    r.queriedUsername.toLowerCase().includes(search.toLowerCase()) ||
    r.resultLabel.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div>
      <div className="page-header">
        <h1>Search History</h1>
        <p>All your previous profile analysis results</p>
      </div>

      {/* Controls */}
      <div style={{display:'flex',gap:'0.75rem',flexWrap:'wrap',marginBottom:'1.5rem',alignItems:'center'}}>
        <div style={{position:'relative',flex:1,minWidth:200}}>
          <Search size={15} style={{position:'absolute',left:'0.75rem',top:'50%',transform:'translateY(-50%)',color:'var(--text-muted)'}}/>
          <input className="form-input" style={{paddingLeft:'2.25rem'}} placeholder="Search by username..." value={search} onChange={e=>setSearch(e.target.value)}/>
        </div>
        <input type="date" className="form-input" style={{width:'auto'}} value={startDate} onChange={e=>setStart(e.target.value)}/>
        <input type="date" className="form-input" style={{width:'auto'}} value={endDate} onChange={e=>setEnd(e.target.value)}/>
        <button className="btn btn-secondary btn-sm" onClick={load}>Filter</button>
        <button className="btn btn-secondary btn-sm" onClick={exportCSV}><Download size={14}/>CSV</button>
        <button className="btn btn-secondary btn-sm" onClick={exportPDF}><FileText size={14}/>PDF</button>
      </div>

      {loading ? <HistorySkeleton/> : filtered.length === 0 ? (
        <div style={{textAlign:'center',padding:'4rem',color:'var(--text-muted)'}}>
          <HistoryIcon size={48} style={{margin:'0 auto 1rem',opacity:0.3}}/>
          <div style={{fontSize:'1.1rem',fontWeight:600}}>No history found</div>
          <div style={{fontSize:'0.875rem',marginTop:'0.25rem'}}>Analyze some profiles to see your history here</div>
        </div>
      ) : (
        <motion.div className="table-wrap" initial={{opacity:0}} animate={{opacity:1}}>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Username</th>
                <th>Result</th>
                <th>Confidence</th>
                <th>Date</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr key={r.historyId}>
                  <td style={{color:'var(--text-muted)',fontSize:'0.8rem'}}>{i+1}</td>
                  <td style={{fontWeight:600}}>@{r.queriedUsername}</td>
                  <td>
                    <span className={`badge badge-${r.resultLabel}`}>
                      {r.resultLabel === 'fake' ? '🚫' : '✅'} {r.resultLabel.toUpperCase()}
                    </span>
                  </td>
                  <td>
                    <div style={{display:'flex',alignItems:'center',gap:'0.5rem'}}>
                      <div className="progress-track" style={{width:60}}>
                        <div className="progress-fill" style={{width:`${Math.round(r.confidenceScore*100)}%`}}/>
                      </div>
                      <span style={{fontSize:'0.8rem',color:'var(--text-secondary)'}}>{Math.round(r.confidenceScore*100)}%</span>
                    </div>
                  </td>
                  <td style={{fontSize:'0.8rem',color:'var(--text-muted)'}}>{r.predictedAt.slice(0,16).replace('T',' ')}</td>
                  <td>
                    <button className="btn btn-danger btn-sm" onClick={()=>deleteRecord(r.historyId)} disabled={deleting===r.historyId}>
                      {deleting===r.historyId ? <span className="spinner" style={{width:12,height:12}}/> : <Trash2 size={13}/>}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      )}
      {!loading && filtered.length > 0 && (
        <div style={{marginTop:'0.75rem',fontSize:'0.8rem',color:'var(--text-muted)'}}>
          Showing {filtered.length} of {records.length} records
        </div>
      )}
    </div>
  )
}
