import { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ShieldCheck, Mail, ArrowLeft, Send } from 'lucide-react'
import api from '../api/client'
import styles from './Auth.module.css'

export default function ForgotPassword() {
  const [email, setEmail]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [done, setDone]         = useState(false)
  const [error, setError]       = useState('')

  const submit = async e => {
    e.preventDefault()
    if (!email) { setError('Email is required.'); return }
    setError('')
    setLoading(true)
    try {
      await api.post('/auth/forgot-password', { email })
      setDone(true)
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong.')
    } finally { setLoading(false) }
  }

  return (
    <div className={styles.page}>
      <div className={styles.bg}/><div className={styles.orb1}/><div className={styles.orb2}/>

      <motion.div className={styles.card}
        initial={{ opacity:0, y:32 }} animate={{ opacity:1, y:0 }}
        transition={{ duration:0.5 }}
      >
        <Link to="/" className={styles.brand}>
          <div className={styles.brandLogo}><ShieldCheck size={20} strokeWidth={2.5}/></div>
          <span>InstaGuard</span>
        </Link>

        <h1 className={styles.title}>Forgot password?</h1>
        <p className={styles.sub}>Enter your email and we'll send you a reset link.</p>

        {error && <div className="alert alert-error" style={{marginBottom:'1rem'}}>{error}</div>}

        {done ? (
          <div className={styles.successBox}>
            <Send size={28}/>
            <div style={{fontWeight:700}}>Check your inbox!</div>
            <div style={{fontSize:'0.85rem', color:'var(--text-secondary)', marginTop:'0.5rem', marginBottom:'1rem'}}>
              If an account exists with that email, we've sent a password reset link. Please check your inbox (and spam folder) and click the link to continue.
            </div>
          </div>
        ) : (
          <form onSubmit={submit} className={styles.form}>
            <div className="form-group">
              <label className="form-label">Email Address</label>
              <div className={styles.inputWrap}>
                <Mail size={16} className={styles.inputIcon}/>
                <input id="forgot-email" type="email" autoComplete="email"
                  className={`form-input ${styles.inputPadded}`}
                  placeholder="you@example.com"
                  value={email} onChange={e => setEmail(e.target.value)} />
              </div>
            </div>
            <button id="forgot-submit" type="submit" className="btn btn-primary btn-full btn-lg" disabled={loading}>
              {loading ? <span className="spinner"/> : <>Send Reset Link <Send size={16}/></>}
            </button>
          </form>
        )}

        <div className={styles.footer}>
          <Link to="/login" className={styles.link}>
            <ArrowLeft size={14} style={{verticalAlign:'middle'}}/> Back to Login
          </Link>
        </div>
      </motion.div>
    </div>
  )
}
