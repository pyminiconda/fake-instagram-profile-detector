import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ShieldCheck, Mail, Lock, User, Eye, EyeOff, ArrowRight, CheckCircle } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'
import styles from './Auth.module.css'

function PasswordStrength({ password }) {
  const checks = [
    { label: 'At least 6 characters', ok: password.length >= 6 },
    { label: 'Contains a number',     ok: /\d/.test(password) },
    { label: 'Contains a letter',     ok: /[a-zA-Z]/.test(password) },
  ]
  const score = checks.filter(c => c.ok).length
  const colors = ['var(--danger)', 'var(--warning)', 'var(--success)']
  const labels = ['Weak', 'Fair', 'Strong']
  if (!password) return null
  return (
    <div style={{ marginTop: '0.4rem' }}>
      <div style={{ display: 'flex', gap: '4px', marginBottom: '0.3rem' }}>
        {[0,1,2].map(i => (
          <div key={i} style={{
            flex:1, height:3, borderRadius:4,
            background: i < score ? colors[score-1] : 'var(--border)',
            transition: 'background 0.3s',
          }}/>
        ))}
      </div>
      {score > 0 && <span style={{fontSize:'0.75rem', color: colors[score-1]}}>{labels[score-1]}</span>}
    </div>
  )
}

export default function Signup() {
  const { signup } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ username:'', email:'', password:'', confirm_password:'' })
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState({})

  const handle = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const validate = () => {
    const errs = {}
    if (!form.username) errs.username = 'Username is required.'
    else if (form.username.length < 3) errs.username = 'At least 3 characters.'
    if (!form.email) errs.email = 'Email is required.'
    if (!form.password) errs.password = 'Password is required.'
    else if (form.password.length < 6) errs.password = 'At least 6 characters.'
    if (form.password !== form.confirm_password) errs.confirm_password = 'Passwords do not match.'
    return errs
  }

  const submit = async e => {
    e.preventDefault()
    const errs = validate()
    if (Object.keys(errs).length) { setErrors(errs); return }
    setErrors({})
    setLoading(true)
    try {
      const user = await signup(form.username, form.email, form.password, form.confirm_password)
      toast.success(`Account created! Welcome, ${user.username} 🎉`)
      navigate('/dashboard')
    } catch (err) {
      const detail = err.response?.data?.detail || 'Signup failed. Please try again.'
      setErrors({ general: detail })
    } finally { setLoading(false) }
  }

  return (
    <div className={styles.page}>
      <div className={styles.bg}/><div className={styles.orb1}/><div className={styles.orb2}/>

      <motion.div className={styles.card}
        initial={{ opacity:0, y:32 }} animate={{ opacity:1, y:0 }}
        transition={{ duration:0.5, ease:[0.4,0,0.2,1] }}
      >
        <Link to="/" className={styles.brand}>
          <div className={styles.brandLogo}><ShieldCheck size={20} strokeWidth={2.5}/></div>
          <span>InstaGuard</span>
        </Link>

        <h1 className={styles.title}>Create an account</h1>
        <p className={styles.sub}>Join InstaGuard and start detecting fake profiles</p>

        {errors.general && <div className="alert alert-error" style={{marginBottom:'1rem'}}>{errors.general}</div>}

        <form onSubmit={submit} className={styles.form}>
          <div className="form-group">
            <label className="form-label">Username</label>
            <div className={styles.inputWrap}>
              <User size={16} className={styles.inputIcon}/>
              <input id="signup-username" name="username" type="text" autoComplete="username"
                className={`form-input ${styles.inputPadded} ${errors.username ? 'error' : ''}`}
                placeholder="your_username" value={form.username} onChange={handle} />
            </div>
            {errors.username && <span className="form-error">{errors.username}</span>}
          </div>

          <div className="form-group">
            <label className="form-label">Email</label>
            <div className={styles.inputWrap}>
              <Mail size={16} className={styles.inputIcon}/>
              <input id="signup-email" name="email" type="email" autoComplete="email"
                className={`form-input ${styles.inputPadded} ${errors.email ? 'error' : ''}`}
                placeholder="you@example.com" value={form.email} onChange={handle} />
            </div>
            {errors.email && <span className="form-error">{errors.email}</span>}
          </div>

          <div className="form-group">
            <label className="form-label">Password</label>
            <div className={styles.inputWrap}>
              <Lock size={16} className={styles.inputIcon}/>
              <input id="signup-password" name="password" type={showPw ? 'text' : 'password'}
                className={`form-input ${styles.inputPadded} ${styles.inputPaddedRight} ${errors.password ? 'error' : ''}`}
                placeholder="Min 6 characters" value={form.password} onChange={handle} />
              <button type="button" className={styles.eyeBtn} onClick={() => setShowPw(p => !p)}>
                {showPw ? <EyeOff size={16}/> : <Eye size={16}/>}
              </button>
            </div>
            <PasswordStrength password={form.password} />
            {errors.password && <span className="form-error">{errors.password}</span>}
          </div>

          <div className="form-group">
            <label className="form-label">Confirm Password</label>
            <div className={styles.inputWrap}>
              <Lock size={16} className={styles.inputIcon}/>
              <input id="signup-confirm" name="confirm_password" type="password"
                className={`form-input ${styles.inputPadded} ${errors.confirm_password ? 'error' : ''}`}
                placeholder="Re-enter password" value={form.confirm_password} onChange={handle} />
            </div>
            {errors.confirm_password && <span className="form-error">{errors.confirm_password}</span>}
          </div>

          <button id="signup-submit" type="submit" className="btn btn-primary btn-full btn-lg" disabled={loading}>
            {loading ? <span className="spinner"/> : <>Create Account <ArrowRight size={18}/></>}
          </button>
        </form>

        <div className={styles.footer}>
          <div>
            Already have an account?{' '}
            <Link to="/login" className={styles.link}>Sign in</Link>
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
