import { Link } from 'react-router-dom'
import { ShieldCheck, Home } from 'lucide-react'

export default function NotFound() {
  return (
    <div style={{ minHeight:'100vh', display:'flex', alignItems:'center', justifyContent:'center', flexDirection:'column', gap:'1.5rem', textAlign:'center', padding:'2rem', background:'var(--bg-base)' }}>
      <div style={{ fontSize:'7rem', fontWeight:900, lineHeight:1, background:'linear-gradient(135deg,var(--brand-primary),var(--brand-secondary))', WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent', backgroundClip:'text' }}>
        404
      </div>
      <div style={{ display:'flex', flexDirection:'column', gap:'0.5rem' }}>
        <h1 style={{ fontSize:'1.5rem', fontWeight:800 }}>Page Not Found</h1>
        <p style={{ color:'var(--text-secondary)' }}>The page you're looking for doesn't exist or has been moved.</p>
      </div>
      <div style={{ display:'flex', gap:'0.75rem' }}>
        <Link to="/" className="btn btn-secondary"><Home size={16}/>Go Home</Link>
        <Link to="/dashboard" className="btn btn-primary"><ShieldCheck size={16}/>Dashboard</Link>
      </div>
    </div>
  )
}
