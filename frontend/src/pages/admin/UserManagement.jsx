import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Users, Plus, Trash2, ShieldCheck, ShieldOff, Search, X, Key } from 'lucide-react'
import api from '../../api/client'
import toast from 'react-hot-toast'
import { useAuth } from '../../context/AuthContext'

function Modal({ title, onClose, children }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <motion.div className="modal" onClick={e => e.stopPropagation()}
        initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'1.25rem' }}>
          <div className="modal-title">{title}</div>
          <button className="btn btn-ghost btn-sm" onClick={onClose}><X size={16}/></button>
        </div>
        {children}
      </motion.div>
    </div>
  )
}

export default function UserManagement() {
  const { user: me } = useAuth()
  const [users, setUsers]     = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch]   = useState('')
  const [showCreate, setShowCreate] = useState(false)
  const [resetTarget, setResetTarget] = useState(null)
  const [deleting, setDeleting]   = useState(null)
  const [toggling, setToggling]   = useState(null)
  const [createForm, setCreateForm] = useState({ username:'', email:'', password:'', is_admin:false })
  const [newPw, setNewPw]     = useState('')
  const [creating, setCreating] = useState(false)

  const load = () => {
    setLoading(true)
    api.get('/admin/users')
      .then(r => { setUsers(r.data.users); setLoading(false) })
      .catch(() => setLoading(false))
  }
  useEffect(() => { load() }, [])

  const createUser = async e => {
    e.preventDefault(); setCreating(true)
    try {
      const { data } = await api.post('/admin/users', createForm)
      setUsers(u => [data, ...u])
      setShowCreate(false)
      setCreateForm({ username:'', email:'', password:'', is_admin:false })
      toast.success(`@${data.username} created!`)
    } catch (err) { toast.error(err.response?.data?.detail || 'Creation failed.') }
    finally { setCreating(false) }
  }

  const deleteUser = async u => {
    if (!window.confirm(`Delete @${u.username}? This cannot be undone.`)) return
    setDeleting(u.userId)
    try {
      await api.delete(`/admin/users/${u.userId}`)
      setUsers(us => us.filter(x => x.userId !== u.userId))
      toast.success(`@${u.username} deleted.`)
    } catch (e) { toast.error(e.response?.data?.detail || 'Delete failed.') }
    finally { setDeleting(null) }
  }

  const toggleRole = async u => {
    setToggling(u.userId)
    try {
      const { data } = await api.put(`/admin/users/${u.userId}/role`, { is_admin: !u.is_admin })
      setUsers(us => us.map(x => x.userId === u.userId ? data : x))
      toast.success(`@${u.username} is now ${data.is_admin ? 'Admin' : 'User'}.`)
    } catch (e) { toast.error(e.response?.data?.detail || 'Role update failed.') }
    finally { setToggling(null) }
  }

  const resetPassword = async () => {
    if (!newPw || newPw.length < 6) { toast.error('Min 6 characters.'); return }
    try {
      await api.put(`/admin/users/${resetTarget.userId}/password`, { new_password: newPw })
      toast.success(`Password reset for @${resetTarget.username}!`)
      setResetTarget(null); setNewPw('')
    } catch { toast.error('Reset failed.') }
  }

  const filtered = users.filter(u =>
    u.username.toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div>
      <div className="page-header">
        <h1>User Management</h1>
        <p>{users.length} registered users</p>
      </div>

      <div style={{ display:'flex', gap:'0.75rem', marginBottom:'1.5rem', flexWrap:'wrap' }}>
        <div style={{ position:'relative', flex:1, minWidth:220 }}>
          <Search size={15} style={{ position:'absolute', left:'0.75rem', top:'50%', transform:'translateY(-50%)', color:'var(--text-muted)' }}/>
          <input className="form-input" style={{ paddingLeft:'2.25rem' }} placeholder="Search users..." value={search} onChange={e => setSearch(e.target.value)}/>
        </div>
        <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
          <Plus size={16}/> Add User
        </button>
      </div>

      {loading ? (
        <div style={{ display:'flex', flexDirection:'column', gap:'0.5rem' }}>
          {[1,2,3,4].map(i => <div key={i} className="skeleton" style={{ height:58, borderRadius:'var(--radius-md)' }}/>)}
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>User</th><th>Email</th><th>Role</th><th>Joined</th><th>Last Login</th><th>Actions</th></tr>
            </thead>
            <tbody>
              {filtered.map(u => (
                <motion.tr key={u.userId} initial={{ opacity:0 }} animate={{ opacity:1 }}>
                  <td>
                    <div style={{ display:'flex', alignItems:'center', gap:'0.65rem' }}>
                      <div style={{ width:32, height:32, borderRadius:'50%', background:'linear-gradient(135deg,var(--brand-primary),#8b5cf6)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.7rem', fontWeight:800, color:'#fff', flexShrink:0 }}>
                        {u.username.slice(0,2).toUpperCase()}
                      </div>
                      <div>
                        <div style={{ fontWeight:600, fontSize:'0.9rem' }}>@{u.username}</div>
                        {u.userId === me?.userId && <div style={{ fontSize:'0.7rem', color:'var(--brand-secondary)' }}>You</div>}
                      </div>
                    </div>
                  </td>
                  <td style={{ fontSize:'0.875rem', color:'var(--text-secondary)' }}>{u.email}</td>
                  <td>
                    <span className={`badge ${u.is_admin ? 'badge-admin' : ''}`}
                      style={!u.is_admin ? { background:'var(--bg-elevated)', color:'var(--text-muted)' } : {}}>
                      {u.is_admin ? '🛡️ Admin' : 'User'}
                    </span>
                  </td>
                  <td style={{ fontSize:'0.8rem', color:'var(--text-muted)' }}>{u.createdAt?.slice(0,10)}</td>
                  <td style={{ fontSize:'0.8rem', color:'var(--text-muted)' }}>{u.lastLogin?.slice(0,10) || '—'}</td>
                  <td>
                    <div style={{ display:'flex', gap:'0.4rem' }}>
                      <button title={u.is_admin ? 'Demote' : 'Promote to Admin'} className="btn btn-ghost btn-sm"
                        disabled={toggling === u.userId || u.userId === me?.userId}
                        onClick={() => toggleRole(u)}>
                        {toggling === u.userId ? <span className="spinner" style={{ width:12, height:12 }}/> : u.is_admin ? <ShieldOff size={14}/> : <ShieldCheck size={14}/>}
                      </button>
                      <button title="Reset Password" className="btn btn-ghost btn-sm" onClick={() => setResetTarget(u)}>
                        <Key size={14}/>
                      </button>
                      <button title="Delete" className="btn btn-danger btn-sm"
                        disabled={deleting === u.userId || u.userId === me?.userId}
                        onClick={() => deleteUser(u)}>
                        {deleting === u.userId ? <span className="spinner" style={{ width:12, height:12 }}/> : <Trash2 size={14}/>}
                      </button>
                    </div>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <div style={{ textAlign:'center', padding:'3rem', color:'var(--text-muted)' }}>
              <Users size={40} style={{ margin:'0 auto 0.75rem', opacity:0.3 }}/> No users found
            </div>
          )}
        </div>
      )}

      {/* Create user modal */}
      {showCreate && (
        <Modal title="Create New User" onClose={() => setShowCreate(false)}>
          <form onSubmit={createUser} style={{ display:'flex', flexDirection:'column', gap:'0.85rem' }}>
            {[['username','Username','text'],['email','Email','email'],['password','Password','password']].map(([k,l,t]) => (
              <div className="form-group" key={k}>
                <label className="form-label">{l}</label>
                <input name={k} type={t} className="form-input" required
                  value={createForm[k]} onChange={e => setCreateForm(f => ({ ...f, [k]: e.target.value }))}/>
              </div>
            ))}
            <label style={{ display:'flex', alignItems:'center', gap:'0.5rem', fontSize:'0.9rem', cursor:'pointer' }}>
              <input type="checkbox" checked={createForm.is_admin}
                onChange={e => setCreateForm(f => ({ ...f, is_admin: e.target.checked }))}/>
              Make Admin
            </label>
            <div style={{ display:'flex', gap:'0.75rem', marginTop:'0.5rem' }}>
              <button type="submit" className="btn btn-primary btn-full" disabled={creating}>
                {creating ? <span className="spinner"/> : 'Create User'}
              </button>
              <button type="button" className="btn btn-ghost btn-full" onClick={() => setShowCreate(false)}>Cancel</button>
            </div>
          </form>
        </Modal>
      )}

      {/* Reset password modal */}
      {resetTarget && (
        <Modal title={`Reset Password — @${resetTarget.username}`} onClose={() => { setResetTarget(null); setNewPw('') }}>
          <input className="form-input" type="password" placeholder="New password (min 6 chars)"
            value={newPw} onChange={e => setNewPw(e.target.value)} style={{ marginBottom:'1rem' }}/>
          <div style={{ display:'flex', gap:'0.75rem' }}>
            <button className="btn btn-primary btn-full" onClick={resetPassword}>Reset Password</button>
            <button className="btn btn-ghost btn-full" onClick={() => { setResetTarget(null); setNewPw('') }}>Cancel</button>
          </div>
        </Modal>
      )}
    </div>
  )
}
