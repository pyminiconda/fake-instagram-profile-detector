import { Link, NavLink, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, Layers, History, User, Settings,
  ShieldCheck, Users, Database, Brain, LogOut, ChevronRight, Moon, Sun
} from 'lucide-react'
import { useAuth } from '../../context/AuthContext'
import { useTheme } from '../../context/ThemeContext'
import toast from 'react-hot-toast'
import styles from './Sidebar.module.css'

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/batch',     icon: Layers,          label: 'Batch Analysis' },
  { to: '/history',   icon: History,         label: 'Search History' },
]

const adminItems = [
  { to: '/admin',         icon: ShieldCheck, label: 'Admin Panel' },
  { to: '/admin/users',   icon: Users,       label: 'User Management' },
  { to: '/admin/dataset', icon: Database,    label: 'Dataset Insights' },
  { to: '/admin/model',   icon: Brain,       label: 'Model Training' },
]

export default function Sidebar() {
  const { user, logout } = useAuth()
  const { theme, toggle } = useTheme()
  const navigate = useNavigate()

  const handleLogout = async () => {
    await logout()
    toast.success('Logged out successfully')
    navigate('/login')
  }

  const initials = user?.username
    ? user.username.slice(0, 2).toUpperCase()
    : 'IG'

  return (
    <aside className={styles.sidebar}>
      {/* Brand */}
      <Link to="/" className={styles.brand} style={{ textDecoration: 'none' }}>
        <div className={styles.logo}>
          <ShieldCheck size={22} strokeWidth={2.5} />
        </div>
        <div>
          <div className={styles.brandName}>InstaGuard</div>
          <div className={styles.brandSub}>Fake Profile Detector</div>
        </div>
      </Link>

      <div className={styles.divider} />

      {/* Main nav */}
      <nav className={styles.nav}>
        <span className={styles.navLabel}>Navigation</span>
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} className={({ isActive }) =>
            `${styles.navItem} ${isActive ? styles.active : ''}`
          }>
            {({ isActive }) => (
              <>
                {isActive && <span className={styles.activePill} />}
                <Icon size={18} />
                <span>{label}</span>
                {isActive && <ChevronRight size={14} className={styles.chevron} />}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Admin nav */}
      {user?.is_admin && (
        <>
          <div className={styles.divider} />
          <nav className={styles.nav}>
            <span className={styles.navLabel}>Admin</span>
            {adminItems.map(({ to, icon: Icon, label }) => (
              <NavLink key={to} to={to} end className={({ isActive }) =>
                `${styles.navItem} ${isActive ? styles.active : ''}`
              }>
                {({ isActive }) => (
                  <>
                    {isActive && <span className={styles.activePill} />}
                    <Icon size={18} />
                    <span>{label}</span>
                    {isActive && <ChevronRight size={14} className={styles.chevron} />}
                  </>
                )}
              </NavLink>
            ))}
          </nav>
        </>
      )}

      <div className={styles.spacer} />
      <div className={styles.divider} />

      {/* Theme toggle */}
      <button className={styles.themeBtn} onClick={toggle} title="Toggle theme">
        {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
      </button>

      {/* User block */}
      <NavLink to="/profile" className={styles.userBlock}>
        <div className={styles.avatar}>{initials}</div>
        <div className={styles.userInfo}>
          <div className={styles.userName}>{user?.username}</div>
          <div className={styles.userRole}>{user?.is_admin ? '🛡️ Admin' : 'User'}</div>
        </div>
        <Settings size={15} className={styles.settingsIcon} />
      </NavLink>

      <button className={styles.logoutBtn} onClick={handleLogout}>
        <LogOut size={16} />
        <span>Logout</span>
      </button>
    </aside>
  )
}
