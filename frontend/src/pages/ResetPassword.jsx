import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ShieldCheck, Lock, KeyRound, CheckCircle } from 'lucide-react'
import api from '../api/client'
import toast from 'react-hot-toast'
import styles from './Auth.module.css'

export default function ResetPassword() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const urlToken = searchParams.get('token') || ''
  
  const [form, setForm]   = useState({ token: urlToken, new_password:'', confirm_password:'' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone]   = useState(false)

  const handle = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const submit = async e => {
    e.preventDefault()
    if (!form.token || !form.new_password || !form.confirm_password) { setError('All fields are required.'); return }
    if (form.new_password !== form.confirm_password) { setError('Passwords do not match.'); return }
    if (form.new_password.length < 6) { setError('Min 6 characters.'); return }
    setError('')
    setLoading(true)
    try {
      await api.post('/auth/reset-password', form)
      setDone(true)
      toast.success('Password reset successfully!')
      setTimeout(() => navigate('/login'), 2500)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to reset password.')
    } finally { setLoading(false) }
  }

  return (
    <div className={styles.page}>
      <div className={styles.bg}/><div className={styles.orb1}/><div className={styles.orb2}/>
      <motion.div className={styles.card} initial={{ opacity:0, y:32 }} animate={{ opacity:1, y:0 }} transition={{ duration:0.5 }}>
        <Link to="/" className={styles.brand}>
          <div className={styles.brandLogo}><ShieldCheck size={20} strokeWidth={2.5}/></div>
          <span>InstaGuard</span>
        </Link>
        <h1 className={styles.title}>Reset password</h1>
        <p className={styles.sub}>Choose a new password for your account.</p>
        {done ? (
          <div className={styles.successBox}>
            <CheckCircle size={32}/>
            <div style={{fontWeight:700}}>Password reset! Redirecting...</div>
          </div>
        ) : (
          <>
            {error && <div className="alert alert-error" style={{marginBottom:'1rem'}}>{error}</div>}
            <form onSubmit={submit} className={styles.form}>
              {!urlToken && (
                <div className="form-group">
                  <label className="form-label">Reset Token</label>
                  <div className={styles.inputWrap}>
                    <KeyRound size={16} className={styles.inputIcon}/>
                    <input id="reset-token" name="token" type="text" className={`form-input ${styles.inputPadded}`} placeholder="Paste token" value={form.token} onChange={handle} />
                  </div>
                </div>
              )}
              <div className="form-group">
                <label className="form-label">New Password</label>
                <div className={styles.inputWrap}>
                  <Lock size={16} className={styles.inputIcon}/>
                  <input id="reset-pw" name="new_password" type="password" className={`form-input ${styles.inputPadded}`} placeholder="Min 6 chars" value={form.new_password} onChange={handle} />
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Confirm Password</label>
                <div className={styles.inputWrap}>
                  <Lock size={16} className={styles.inputIcon}/>
                  <input id="reset-confirm" name="confirm_password" type="password" className={`form-input ${styles.inputPadded}`} placeholder="Repeat password" value={form.confirm_password} onChange={handle} />
                </div>
              </div>
              <button id="reset-submit" type="submit" className="btn btn-primary btn-full btn-lg" disabled={loading}>
                {loading ? <span className="spinner"/> : 'Reset Password'}
              </button>
            </form>
          </>
        )}
        <div className={styles.footer}><Link to="/login" className={styles.link}>Back to Login</Link></div>
      </motion.div>
    </div>
  )
}
