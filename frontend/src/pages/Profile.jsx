import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { User, Mail, Lock, Trash2, Save, Edit2, AlertTriangle } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import api from '../api/client'
import toast from 'react-hot-toast'

function Section({ title, icon: Icon, children }) {
  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div style={{ display:'flex', alignItems:'center', gap:'0.6rem', marginBottom:'1.25rem' }}>
        <div style={{ width:36,height:36,borderRadius:'var(--radius-md)',background:'rgba(108,99,255,0.12)',display:'flex',alignItems:'center',justifyContent:'center',color:'var(--brand-secondary)' }}>
          <Icon size={18}/>
        </div>
        <h2 style={{ fontSize:'1rem', fontWeight:700 }}>{title}</h2>
      </div>
      {children}
    </div>
  )
}

export default function Profile() {
  const { user, refreshUser, logout } = useAuth()
  const [stats, setStats] = useState(null)
  const [username, setUsername] = useState(user?.username || '')
  const [email, setEmail]       = useState(user?.email || '')
  const [emailPw, setEmailPw]   = useState('')
  const [pwForm, setPwForm]     = useState({ current_password:'', new_password:'', confirm_password:'' })
  const [delPw, setDelPw]       = useState('')
  const [showDelModal, setShowDel] = useState(false)
  const [saving, setSaving]     = useState({})

  useEffect(() => {
    api.get('/users/me/stats').then(r => setStats(r.data)).catch(() => {})
  }, [])

  const save = async (key, fn) => {
    setSaving(s => ({ ...s, [key]: true }))
    try {
      await fn()
      await refreshUser()
      toast.success('Updated successfully!')
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Update failed.')
    } finally { setSaving(s => ({ ...s, [key]: false })) }
  }

  const deleteAccount = async () => {
    try {
      await api.delete('/users/me', { data: { password: delPw } })
      toast.success('Account deleted.')
      await logout()
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Incorrect password.')
    }
  }

  const initials = user?.username ? user.username.slice(0,2).toUpperCase() : 'IG'

  return (
    <div>
      <div className="page-header">
        <h1>My Profile</h1>
        <p>Manage your account information and preferences</p>
      </div>

      {/* Avatar card */}
      <div className="card" style={{display:'flex',alignItems:'center',gap:'1.5rem',marginBottom:'1.25rem'}}>
        <div style={{width:72,height:72,borderRadius:'50%',background:'linear-gradient(135deg,var(--brand-primary),#8b5cf6)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:'1.5rem',fontWeight:800,color:'#fff',flexShrink:0}}>
          {initials}
        </div>
        <div>
          <div style={{fontSize:'1.3rem',fontWeight:800}}>{user?.username}</div>
          <div style={{color:'var(--text-secondary)',fontSize:'0.875rem'}}>{user?.email}</div>
          <div style={{display:'flex',gap:'0.5rem',marginTop:'0.5rem'}}>
            {user?.is_admin && <span className="badge badge-admin">🛡️ Admin</span>}
            <span style={{fontSize:'0.75rem',color:'var(--text-muted)'}}>Member since {user?.createdAt?.slice(0,10)}</span>
          </div>
        </div>
        {stats && (
          <div style={{marginLeft:'auto',display:'flex',gap:'1.5rem',textAlign:'center'}}>
            {[['Total Scans',stats.total_scans],['Fake Found',stats.fake_found],['Genuine',stats.genuine_found]].map(([l,v])=>(
              <div key={l}>
                <div style={{fontSize:'1.6rem',fontWeight:800,color:'var(--brand-primary)'}}>{v}</div>
                <div style={{fontSize:'0.75rem',color:'var(--text-muted)'}}>{l}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Username */}
      <Section title="Change Username" icon={User}>
        <div style={{display:'flex',gap:'0.75rem',flexWrap:'wrap'}}>
          <input className="form-input" style={{flex:1}} value={username} onChange={e=>setUsername(e.target.value)} placeholder="New username"/>
          <button className="btn btn-primary" disabled={saving.username||username===user?.username} onClick={()=>save('username',()=>api.put('/users/me/username',{username}))}>
            {saving.username ? <span className="spinner"/> : <><Save size={15}/>Save</>}
          </button>
        </div>
      </Section>

      {/* Email */}
      <Section title="Change Email" icon={Mail}>
        <div style={{display:'flex',flexDirection:'column',gap:'0.75rem'}}>
          <input className="form-input" value={email} onChange={e=>setEmail(e.target.value)} placeholder="New email address" type="email"/>
          <input className="form-input" value={emailPw} onChange={e=>setEmailPw(e.target.value)} placeholder="Current password to confirm" type="password"/>
          <button className="btn btn-primary" style={{alignSelf:'flex-start'}} disabled={saving.email} onClick={()=>save('email',()=>api.put('/users/me/email',{email,current_password:emailPw}))}>
            {saving.email ? <span className="spinner"/> : <><Save size={15}/>Update Email</>}
          </button>
        </div>
      </Section>

      {/* Password */}
      <Section title="Change Password" icon={Lock}>
        <div style={{display:'flex',flexDirection:'column',gap:'0.75rem',maxWidth:400}}>
          {[['current_password','Current Password'],['new_password','New Password'],['confirm_password','Confirm New Password']].map(([k,l])=>(
            <input key={k} className="form-input" type="password" placeholder={l} value={pwForm[k]} onChange={e=>setPwForm(f=>({...f,[k]:e.target.value}))}/>
          ))}
          <button className="btn btn-primary" style={{alignSelf:'flex-start'}} disabled={saving.pw} onClick={()=>save('pw',()=>api.put('/users/me/password',pwForm))}>
            {saving.pw ? <span className="spinner"/> : 'Update Password'}
          </button>
        </div>
      </Section>

      {/* Delete Account */}
      <Section title="Danger Zone" icon={Trash2}>
        <div className="alert alert-error" style={{marginBottom:'1rem'}}>
          <AlertTriangle size={18}/> Deleting your account is permanent and cannot be undone.
        </div>
        <button className="btn btn-danger" onClick={()=>setShowDel(true)}>
          <Trash2 size={15}/> Delete My Account
        </button>
      </Section>

      {/* Delete Modal */}
      {showDelModal && (
        <div className="modal-backdrop" onClick={()=>setShowDel(false)}>
          <motion.div className="modal" onClick={e=>e.stopPropagation()} initial={{scale:0.9,opacity:0}} animate={{scale:1,opacity:1}}>
            <div className="modal-title" style={{color:'var(--danger)'}}>⚠️ Delete Account</div>
            <p style={{color:'var(--text-secondary)',fontSize:'0.9rem',marginBottom:'1rem'}}>
              This will permanently delete your account and all your history. Enter your password to confirm.
            </p>
            <input className="form-input" type="password" placeholder="Enter your password" value={delPw} onChange={e=>setDelPw(e.target.value)} style={{marginBottom:'1rem'}}/>
            <div style={{display:'flex',gap:'0.75rem'}}>
              <button className="btn btn-danger btn-full" onClick={deleteAccount}>Yes, Delete My Account</button>
              <button className="btn btn-ghost btn-full" onClick={()=>setShowDel(false)}>Cancel</button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}
