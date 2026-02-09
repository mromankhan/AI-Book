import React, { useState } from 'react';
import { useAuth } from '../Auth/AuthProvider';
import styles from './chapterActions.module.css';

const API_BASE = typeof window !== 'undefined'
  ? (window as any).__CHATBOT_API_URL || 'http://localhost:8000'
  : 'http://localhost:8000';

interface PersonalizeButtonProps {
  chapter: string;
  onPersonalized: (content: string) => void;
  isActive: boolean;
  onReset: () => void;
}

export default function PersonalizeButton({ chapter, onPersonalized, isActive, onReset }: PersonalizeButtonProps) {
  const { user, token } = useAuth();
  const [loading, setLoading] = useState(false);

  const handlePersonalize = async () => {
    if (isActive) {
      onReset();
      return;
    }

    if (!user || !token) {
      alert('Please sign in to personalize content.');
      return;
    }

    setLoading(true);
    try {
      // Get current page content
      const mainContent = document.querySelector('article')?.textContent || '';

      const res = await fetch(`${API_BASE}/api/personalize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ chapter, content: mainContent.slice(0, 8000) }),
      });

      if (res.ok) {
        const data = await res.json();
        onPersonalized(data.personalized_content);
      } else {
        alert('Personalization failed. Please try again.');
      }
    } catch {
      alert('Could not connect to the server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      className={`${styles.actionButton} ${isActive ? styles.activeButton : ''}`}
      onClick={handlePersonalize}
      disabled={loading}
    >
      {loading ? (
        <><span className={styles.loadingSpinner} /> Personalizing...</>
      ) : isActive ? (
        'Show Original'
      ) : (
        'Personalize Content'
      )}
    </button>
  );
}
