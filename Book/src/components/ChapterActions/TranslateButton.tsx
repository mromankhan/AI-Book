import React, { useState } from 'react';
import styles from './chapterActions.module.css';

const API_BASE = typeof window !== 'undefined'
  ? (window as any).__CHATBOT_API_URL || 'http://localhost:8000'
  : 'http://localhost:8000';

interface TranslateButtonProps {
  chapter: string;
  onTranslated: (content: string) => void;
  isActive: boolean;
  onReset: () => void;
}

export default function TranslateButton({ chapter, onTranslated, isActive, onReset }: TranslateButtonProps) {
  const [loading, setLoading] = useState(false);

  const handleTranslate = async () => {
    if (isActive) {
      onReset();
      return;
    }

    setLoading(true);
    try {
      // Get current page content
      const mainContent = document.querySelector('article')?.textContent || '';

      const res = await fetch(`${API_BASE}/api/translate/urdu`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chapter, content: mainContent.slice(0, 8000) }),
      });

      if (res.ok) {
        const data = await res.json();
        onTranslated(data.translated_content);
      } else {
        alert('Translation failed. Please try again.');
      }
    } catch {
      alert('Could not connect to the server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      className={`${styles.actionButton} ${styles.urduButton} ${isActive ? styles.activeButton : ''}`}
      onClick={handleTranslate}
      disabled={loading}
    >
      {loading ? (
        <><span className={styles.loadingSpinner} /> Translating...</>
      ) : isActive ? (
        'Show English'
      ) : (
        'اردو میں ترجمہ'
      )}
    </button>
  );
}
