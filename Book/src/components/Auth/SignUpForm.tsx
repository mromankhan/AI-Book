import React, { useState } from 'react';
import { useAuth } from './AuthProvider';
import styles from './auth.module.css';

interface SignUpFormProps {
  onClose: () => void;
  onSwitchToSignIn: () => void;
}

export default function SignUpForm({ onClose, onSwitchToSignIn }: SignUpFormProps) {
  const { signUp } = useAuth();
  const [form, setForm] = useState({
    name: '',
    email: '',
    password: '',
    programming_level: 'beginner',
    hardware_experience: 'none',
    education_level: 'undergraduate',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signUp({ ...form, interests: [] });
      onClose();
    } catch (err: any) {
      setError(err.message || 'Signup failed');
    } finally {
      setLoading(false);
    }
  };

  const update = (field: string, value: string) =>
    setForm(prev => ({ ...prev, [field]: value }));

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <h2 className={styles.modalTitle}>Create Account</h2>
        <form className={styles.form} onSubmit={handleSubmit}>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>Name</label>
            <input className={styles.input} type="text" required value={form.name}
              onChange={e => update('name', e.target.value)} placeholder="Your name" />
          </div>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>Email</label>
            <input className={styles.input} type="email" required value={form.email}
              onChange={e => update('email', e.target.value)} placeholder="email@example.com" />
          </div>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>Password</label>
            <input className={styles.input} type="password" required minLength={6} value={form.password}
              onChange={e => update('password', e.target.value)} placeholder="Min 6 characters" />
          </div>

          <p className={styles.sectionTitle}>Tell us about your background</p>

          <div className={styles.fieldGroup}>
            <label className={styles.label}>Programming Experience</label>
            <select className={styles.select} value={form.programming_level}
              onChange={e => update('programming_level', e.target.value)}>
              <option value="beginner">Beginner - Just starting out</option>
              <option value="intermediate">Intermediate - Some projects done</option>
              <option value="advanced">Advanced - Professional experience</option>
            </select>
          </div>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>Hardware / Robotics Experience</label>
            <select className={styles.select} value={form.hardware_experience}
              onChange={e => update('hardware_experience', e.target.value)}>
              <option value="none">None - Purely software background</option>
              <option value="some">Some - Arduino/Raspberry Pi projects</option>
              <option value="extensive">Extensive - Worked with robots/embedded systems</option>
            </select>
          </div>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>Education Level</label>
            <select className={styles.select} value={form.education_level}
              onChange={e => update('education_level', e.target.value)}>
              <option value="high_school">High School</option>
              <option value="undergraduate">Undergraduate</option>
              <option value="graduate">Graduate / Postgraduate</option>
              <option value="professional">Working Professional</option>
            </select>
          </div>

          {error && <p className={styles.error}>{error}</p>}

          <button className={styles.submitButton} type="submit" disabled={loading}>
            {loading ? 'Creating account...' : 'Sign Up'}
          </button>
        </form>
        <p className={styles.switchText}>
          Already have an account?{' '}
          <span className={styles.switchLink} onClick={onSwitchToSignIn}>Sign In</span>
        </p>
      </div>
    </div>
  );
}
