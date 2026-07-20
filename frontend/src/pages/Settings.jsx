import { Moon, Sun, Monitor, Bell, Database, Info, Palette } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'
import toast from 'react-hot-toast'
import api from '../api/client'

function SettingRow({ title, desc, children }) {
  return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'1rem 0', borderBottom:'1px solid var(--border)', gap:'1rem' }}>
      <div>
        <div style={{ fontWeight:600, fontSize:'0.9rem' }}>{title}</div>
        {desc && <div style={{ fontSize:'0.8rem', color:'var(--text-muted)', marginTop:'0.2rem' }}>{desc}</div>}
      </div>
      <div style={{ flexShrink:0 }}>{children}</div>
    </div>
  )
}

export default function Settings() {
  const { theme, setTheme } = useTheme()

  const exportAllData = async () => {
    try {
      const res = await api.get('/history/export/csv', { responseType:'blob' })
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a'); a.href=url; a.download='my_instaGuard_data.csv'; a.click()
      URL.revokeObjectURL(url)
      toast.success('Data exported!')
    } catch { toast.error('No data to export.') }
  }

  const themes = [
    { value:'dark',  icon: Moon,    label:'Dark' },
    { value:'light', icon: Sun,     label:'Light' },
    { value:'system',icon: Monitor, label:'System' },
  ]

  return (
    <div>
      <div className="page-header">
        <h1>Settings</h1>
        <p>Customize your InstaGuard experience</p>
      </div>

      {/* Appearance */}
      <div className="card" style={{ marginBottom:'1.25rem' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'0.6rem', marginBottom:'1.25rem' }}>
          <div style={{ width:36,height:36,borderRadius:'var(--radius-md)',background:'rgba(108,99,255,0.12)',display:'flex',alignItems:'center',justifyContent:'center',color:'var(--brand-secondary)' }}>
            <Palette size={18}/>
          </div>
          <h2 style={{ fontSize:'1rem', fontWeight:700 }}>Appearance</h2>
        </div>

        <SettingRow title="Color Theme" desc="Choose how InstaGuard looks to you">
          <div style={{ display:'flex', gap:'0.5rem' }}>
            {themes.map(({ value, icon: Icon, label }) => (
              <button key={value} onClick={() => {
                if (value === 'system') {
                  const sys = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
                  setTheme(sys)
                } else { setTheme(value) }
                toast.success(`${label} theme applied!`)
              }}
                className={`btn btn-sm ${(theme===value||(value==='system'&&false)) ? 'btn-primary' : 'btn-secondary'}`}
                style={{ minWidth:72 }}
              >
                <Icon size={14}/> {label}
              </button>
            ))}
          </div>
        </SettingRow>

        <SettingRow title="Current Theme" desc="Your active color scheme">
          <span className="badge" style={{ background:'var(--brand-glow)', color:'var(--brand-secondary)' }}>
            {theme === 'dark' ? '🌙 Dark' : '☀️ Light'}
          </span>
        </SettingRow>
      </div>

      {/* Data & Privacy */}
      <div className="card" style={{ marginBottom:'1.25rem' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'0.6rem', marginBottom:'1.25rem' }}>
          <div style={{ width:36,height:36,borderRadius:'var(--radius-md)',background:'rgba(108,99,255,0.12)',display:'flex',alignItems:'center',justifyContent:'center',color:'var(--brand-secondary)' }}>
            <Database size={18}/>
          </div>
          <h2 style={{ fontSize:'1rem', fontWeight:700 }}>Data & Privacy</h2>
        </div>

        <SettingRow title="Export My Data" desc="Download all your search history as CSV">
          <button className="btn btn-secondary btn-sm" onClick={exportAllData}>
            <Database size={14}/> Export Data
          </button>
        </SettingRow>

        <SettingRow title="Data Storage" desc="All your data is stored locally on our server — never shared with third parties">
          <span className="badge badge-genuine">🔒 Local Only</span>
        </SettingRow>
      </div>

      {/* About */}
      <div className="card">
        <div style={{ display:'flex', alignItems:'center', gap:'0.6rem', marginBottom:'1.25rem' }}>
          <div style={{ width:36,height:36,borderRadius:'var(--radius-md)',background:'rgba(108,99,255,0.12)',display:'flex',alignItems:'center',justifyContent:'center',color:'var(--brand-secondary)' }}>
            <Info size={18}/>
          </div>
          <h2 style={{ fontSize:'1rem', fontWeight:700 }}>About InstaGuard</h2>
        </div>

        {[
          ['Application', 'InstaGuard — Fake Instagram Profile Detector'],
          ['Version', '1.0.0'],
          ['ML Model', 'XGBoost / Random Forest with SHAP Explainability'],
          ['Backend', 'FastAPI + Python'],
          ['Frontend', 'React 19 + Vite'],
          ['Authors', 'Muhammad Moiz Nasim & Muhammad Awais'],
        ].map(([k,v]) => (
          <SettingRow key={k} title={k} desc={null}>
            <span style={{ fontSize:'0.85rem', color:'var(--text-secondary)' }}>{v}</span>
          </SettingRow>
        ))}
      </div>
    </div>
  )
}
