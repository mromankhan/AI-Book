import React, { useState } from 'react';
import { useAuth } from './AuthProvider';
import styles from './auth.module.css';

interface SignInFormProps {
  onClose: () => void;
  onSwitchToSignUp: () => void;
}

export default function SignInForm({ onClose, onSwitchToSignUp }: SignInFormProps) {
  const { signIn } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signIn(email, password);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Sign in failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <h2 className={styles.modalTitle}>Sign In</h2>
        <form className={styles.form} onSubmit={handleSubmit}>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>Email</label>
            <input className={styles.input} type="email" required value={email}
              onChange={e => setEmail(e.target.value)} placeholder="email@example.com" />
          </div>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>Password</label>
            <input className={styles.input} type="password" required value={password}
              onChange={e => setPassword(e.target.value)} placeholder="Your password" />
          </div>

          {error && <p className={styles.error}>{error}</p>}

          <button className={styles.submitButton} type="submit" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
        <p className={styles.switchText}>
          Don't have an account?{' '}
          <span className={styles.switchLink} onClick={onSwitchToSignUp}>Sign Up</span>
        </p>
      </div>
    </div>
  );
}
