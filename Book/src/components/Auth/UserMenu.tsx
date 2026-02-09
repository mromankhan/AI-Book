import React, { useState } from 'react';
import { useAuth } from './AuthProvider';
import SignUpForm from './SignUpForm';
import SignInForm from './SignInForm';
import styles from './auth.module.css';

export default function UserMenu() {
  const { user, signOut } = useAuth();
  const [showSignUp, setShowSignUp] = useState(false);
  const [showSignIn, setShowSignIn] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  if (user) {
    return (
      <div className={styles.userMenu} style={{ position: 'relative' }}>
        <button className={styles.userButton} onClick={() => setShowDropdown(!showDropdown)}>
          {user.name || user.email}
        </button>
        {showDropdown && (
          <div className={styles.dropdown}>
            <div style={{ padding: '8px 14px', fontSize: 12, color: 'var(--ifm-color-emphasis-500)' }}>
              {user.email}
            </div>
            <button className={styles.dropdownItem} onClick={() => { signOut(); setShowDropdown(false); }}>
              Sign Out
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      <button className={styles.authButton} onClick={() => setShowSignIn(true)}>
        Sign In
      </button>
      {showSignIn && (
        <SignInForm
          onClose={() => setShowSignIn(false)}
          onSwitchToSignUp={() => { setShowSignIn(false); setShowSignUp(true); }}
        />
      )}
      {showSignUp && (
        <SignUpForm
          onClose={() => setShowSignUp(false)}
          onSwitchToSignIn={() => { setShowSignUp(false); setShowSignIn(true); }}
        />
      )}
    </>
  );
}
