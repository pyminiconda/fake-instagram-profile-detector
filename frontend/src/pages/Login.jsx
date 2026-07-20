import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ShieldCheck, Mail, Lock, Eye, EyeOff, ArrowRight } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'
import styles from './Auth.module.css'

export default function Login() {
  const { login } = useAuth()
  const navigate   = useNavigate()
  const [form, setForm]     = useState({ email: '', password: '' })
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState('')

  const handle = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const submit = async e => {
    e.preventDefault()
    setError('')
    if (!form.email || !form.password) { setError('All fields are required.'); return }
    setLoading(true)
    try {
      const user = await login(form.email, form.password)
      toast.success(`Welcome back, ${user.username}! 👋`)
      navigate('/dashboard')
    } catch (err) {
      setError(err.response?.data?.detail || 'Invalid email or password.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.bg} /><div className={styles.orb1}/><div className={styles.orb2}/>

      <motion.div className={styles.card}
        initial={{ opacity: 0, y: 32 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.4,0,0.2,1] }}
      >
        <Link to="/" className={styles.brand}>
          <div className={styles.brandLogo}><ShieldCheck size={20} strokeWidth={2.5}/></div>
          <span>InstaGuard</span>
        </Link>

        <h1 className={styles.title}>Welcome back</h1>
        <p className={styles.sub}>Sign in to your account to continue</p>

        {error && <div className="alert alert-error" style={{marginBottom:'1rem'}}>{error}</div>}

        <form onSubmit={submit} className={styles.form}>
          <div className="form-group">
            <label className="form-label">Email</label>
            <div className={styles.inputWrap}>
              <Mail size={16} className={styles.inputIcon}/>
              <input id="login-email" name="email" type="email" autoComplete="email"
                className={`form-input ${styles.inputPadded}`}
                placeholder="you@example.com" value={form.email} onChange={handle} />
            </div>
          </div>

          <div className="form-group">
            <div className={styles.labelRow}>
              <label className="form-label">Password</label>
              <Link to="/forgot-password" className={styles.forgotLink}>Forgot password?</Link>
            </div>
            <div className={styles.inputWrap}>
              <Lock size={16} className={styles.inputIcon}/>
              <input id="login-password" name="password" type={showPw ? 'text' : 'password'} autoComplete="current-password"
                className={`form-input ${styles.inputPadded} ${styles.inputPaddedRight}`}
                placeholder="••••••••" value={form.password} onChange={handle} />
              <button type="button" className={styles.eyeBtn} onClick={() => setShowPw(p => !p)}>
                {showPw ? <EyeOff size={16}/> : <Eye size={16}/>}
              </button>
            </div>
          </div>

          <button id="login-submit" type="submit" className="btn btn-primary btn-full btn-lg" disabled={loading}>
            {loading ? <span className="spinner"/> : <>Sign In <ArrowRight size={18}/></>}
          </button>
        </form>

        <div className={styles.footer}>
          <div>
            Don't have an account?{' '}
            <Link to="/signup" className={styles.link}>Create one free</Link>
          </div>
          <div style={{ marginTop: '0.75rem' }}>
            <Link to="/" className={styles.link} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', opacity: 0.75 }}>
              ← Return to home page
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
